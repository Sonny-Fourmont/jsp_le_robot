#!/usr/bin/env bash
# run.sh - one command to start the SO-ARM101 stack in a given mode.
#
#   ./run.sh cube          simulation phase 1 (cube + boite) : conteneurs, RViz, shell ROS
#   ./run.sh ouija         simulation phase 2 (planche Ouija + ollama) : idem
#   ./run.sh real          vrai bras + camera USB, avec les valeurs du .env
#   ./run.sh video [N] [main|rviz|grid]  simu cube + enregistre N pick & place dans ros2_ws/rec/pick_place_sim.mp4
#       grid = 1920x1080, scene MuJoCo + RViz (TF, FK) + camera + detection : ce que demande le prof
#                          main : vue 3D MuJoCo seule (pick_place_sim_main.mp4)
#                          rviz : vue RViz (TF + FK du TCP, pick_place_rviz.mp4), rviz2 tourne sans ecran dans le conteneur
#   ./run.sh video-ouija ["question"] [REPONSE]  simu ouija + video de la reponse epelee dans ros2_ws/rec/ouija_demo.mp4
#   ./run.sh video-reel ["question"] [REPONSE|LLM]  VRAI ROBOT (valeurs du .env) : video 1920x1080 RViz + camera +
#                          detection pendant que le bras epelle REPONSE (LLM : la question passe par ollama)
#                          -> ros2_ws/rec/ouija_reel.mp4. Filmer la scene au telephone en meme temps.
#   ./run.sh hsv           regle les seuils HSV sur la vraie camera (fenetre, curseurs, s = sauver)
#   ./run.sh rviz          ouvre RViz sur la stack en cours
#   ./run.sh shell         shell ROS dans le conteneur control
#   ./run.sh stop          arrete tous les conteneurs
#   ./run.sh start <mode>  lance et attend sans ouvrir RViz ni shell
#
# Les modes simu passent USE_SIM / SIM_SCENE / CAMERA_TF / ARM_MAX_SPEED au lancement
# (priorite sur le .env, qui peut rester regle pour le vrai bras). Pour forcer la
# calibration ArUco en simu : CAMERA_TF=calib ./run.sh cube
set -euo pipefail
cd "$(dirname "$0")"

ROS_SRC="source /opt/ros/jazzy/setup.bash && source /ros2_ws/install/setup.bash"

# Simulation: scene geometry and built-in thresholds, whatever the .env says for the real robot.
sim_env() {
    export USE_SIM=true CAMERA_TF="${CAMERA_TF:-sim}" ARM_MAX_SPEED=1.0 CALIB_MODE=calibrate
    export CALIB_SHEET_POSE="-0.15 0.20 0.0" OUIJA_BOARD_POSE="0.235 0.0 0.0" PERCEPTION_PARAMS=
}

set_mode() {
    case "$1" in
        cube)  sim_env; export SIM_SCENE=
               SERVICES="control viz perception" ;;
        ouija) sim_env; export SIM_SCENE=scene_ouija.xml
               SERVICES="control viz perception ollama" ;;
        real)  export USE_SIM=false
               SERVICES="control viz perception camera"; PROFILE="--profile real" ;;
        *) echo "mode inconnu : $1 (cube | ouija | real)"; exit 1 ;;
    esac
}

start() {
    set_mode "$1"
    echo ">> mode $1 : USE_SIM=${USE_SIM} SIM_SCENE=${SIM_SCENE:-scene_tek5.xml (defaut)} CAMERA_TF=${CAMERA_TF:-<.env>}"
    local since; since=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    # perception already running and the scene changes (mode switch): restart it so camera_calib
    # re-estimates the camera pose in the new scene (the Ouija scene has its own camera)
    local restart_perception="" cur_scene="" cid
    cid=$(docker compose ps -q control 2>/dev/null || true)
    if [ -n "$cid" ]; then  # (set -e: a failing command substitution would exit the script)
        cur_scene=$( { docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$cid" 2>/dev/null || true; } | sed -n 's/^SIM_SCENE=//p')
    fi
    if docker compose ps --status running --services 2>/dev/null | grep -qx perception && [ "$cur_scene" != "${SIM_SCENE:-}" ]; then
        restart_perception=1
    fi
    # shellcheck disable=SC2086
    docker compose ${PROFILE:-} up -d $SERVICES
    [ -n "$restart_perception" ] && docker compose restart perception >/dev/null 2>&1
    echo ">> attente de 'Brain node ready.' (compilation du workspace au premier demarrage)..."
    for _ in $(seq 1 180); do
        if docker compose logs --since "$since" control 2>/dev/null | grep -q "Brain node ready."; then
            echo ">> pret."; wait_perception; return 0
        fi
        # already running before this call (no new 'ready' line): the brain services are up
        if docker compose exec -T control bash -lc "$ROS_SRC && timeout 5 ros2 service list 2>/dev/null" 2>/dev/null \
                | grep -qE "^/(go_to_target|ouija/spell)$"; then
            echo ">> deja pret."; wait_perception; return 0
        fi
        # a node that died at startup (exit code -2 = SIGINT of a previous shutdown, ignored)
        if docker compose logs --since "$since" control 2>/dev/null | grep "process has died" | grep -qv "exit code -2"; then
            echo "!! un noeud s'est arrete au demarrage, voir : docker compose logs control"; return 1
        fi
        sleep 1
    done
    echo "!! toujours pas pret apres 3 min, voir : docker compose logs control"; return 1
}

# The perception node must publish its debug image (= it receives the camera images and the
# camera TF). Seen once: node alive but receiving nothing after a restart of control; a restart
# of perception fixes it.
wait_perception() {
    local topic=/perception/debug_image
    [ "${SIM_SCENE:-}" = scene_ouija.xml ] && topic=/ouija/debug_image
    for attempt in 1 2; do
        if docker compose exec -T control bash -lc "$ROS_SRC && python3 /ros2_ws/tools/wait_topic.py $topic 25" >/dev/null 2>&1; then
            return 0
        fi
        if [ "$attempt" = 1 ]; then
            echo ">> perception muette ($topic) : redemarrage de perception..."
            docker compose restart perception >/dev/null 2>&1
        fi
    done
    echo "!! perception toujours muette, voir : docker compose logs perception"; return 1
}

rviz() {
    xhost +local:docker >/dev/null 2>&1 || true
    docker compose exec -d control bash -lc "$ROS_SRC && rviz2 -d /ros2_ws/so101.rviz"
    echo ">> RViz lance."
}

shell() {
    cat <<'TXT'
>> shell ROS. Commandes utiles :
   ros2 node list && ros2 topic list
   ros2 service call /set_joint_positions so101_interfaces/srv/SetJointPositions "{positions: [0, -80, 80, 60, 0, 100]}"
   ros2 service call /go_to_target so101_interfaces/srv/GoToTarget "{target_pose: {header: {frame_id: world}, pose: {position: {x: 0.25, y: 0.05, z: 0.08}}}}"
   ros2 service call /pick_and_place std_srvs/srv/Trigger
   ros2 service call /sim/randomize_object std_srvs/srv/Trigger
   ros2 service call /ouija/ask so101_interfaces/srv/Ask "{question: 'est-ce que tu es vivant ?'}"
TXT
    exec docker compose exec control bash -c "$ROS_SRC && exec bash -i"
}

case "${1:-}" in
    cube|ouija|real) start "$1"; rviz; shell ;;
    start) start "${2:-cube}" ;;
    video) start cube
           name=pick_place_sim; layout=""
           [ "${3:-}" = main ] && { name=pick_place_sim_main; layout=--main; }
           [ "${3:-}" = rviz ] && { name=pick_place_rviz; layout=--rviz; }
           [ "${3:-}" = grid ] && { name=pick_place_grid; layout=--grid; }
           docker compose exec control bash -lc "$ROS_SRC && python3 /ros2_ws/tools/record_demo.py /ros2_ws/rec/$name.mp4 ${2:-3} $layout"
           echo ">> video : $(cd .. && pwd)/ros2_ws/rec/$name.mp4" ;;
    video-reel) start real
           [ "${3:-}" = LLM ] && docker compose up -d ollama >/dev/null 2>&1
           flag=""; [ "${3:-}" = LLM ] && flag=--ask
           docker compose exec control bash -lc "$ROS_SRC && python3 /ros2_ws/tools/record_ouija.py /ros2_ws/rec/ouija_reel.mp4 \"${2:-Qui est le meilleur groupe ?}\" \"${3:-NOUS}\" --real $flag"
           echo ">> video : $(cd .. && pwd)/ros2_ws/rec/ouija_reel.mp4" ;;
    video-ouija) start ouija
           docker compose exec control bash -lc "$ROS_SRC && python3 /ros2_ws/tools/record_ouija.py /ros2_ws/rec/ouija_demo.mp4 \"${2:-Qui est le meilleur groupe ?}\" \"${3:-NOUS}\""
           echo ">> video : $(cd .. && pwd)/ros2_ws/rec/ouija_demo.mp4" ;;
    rviz) rviz ;;
    hsv) xhost +local:docker >/dev/null 2>&1 || true
         docker compose --profile real run --rm --no-deps camera bash -lc "source /opt/ros/jazzy/setup.bash && python3 /ros2_ws/tools/hsv_tune.py" ;;
    shell) shell ;;
    stop) docker compose --profile real stop ;;
    *) sed -n '2,14p' "$0" ;;
esac
