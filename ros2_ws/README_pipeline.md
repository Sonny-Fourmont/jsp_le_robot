# Pipeline SO-ARM101 : état et mode d'emploi (équipe)

Tout tourne en simulation (`USE_SIM=true` dans `docker/.env`). Trois conteneurs :
`control` (driver + brain + ouija), `viz` (robot_state_publisher, TF), `perception`
(caméra -> 3D : cube/boîte de la phase 1, planche Ouija + planchette de la phase 2).

Deux scènes MuJoCo, choisies par `SIM_SCENE` dans `docker/.env` :
`SIM_SCENE=` (vide) = scène phase 1 (cube + boîte), `SIM_SCENE=scene_ouija.xml` = phase 2.
Changer de scène = `docker compose up -d control` (recrée le conteneur).

## Lancer

```bash
cd docker
cp .env.example .env            # une fois ; ROS_DOMAIN_ID propre au groupe
docker compose build            # une fois (control + perception)
xhost +local:docker             # à chaque session graphique
docker compose up -d control viz perception
docker compose logs -f control  # attendre "Driver node ready." et "Brain node ready."
# RViz
docker compose exec control bash -lc "source /ros2_ws/install/setup.bash && rviz2 -d /ros2_ws/so101.rviz"
# un shell ROS
docker compose exec control bash
source /opt/ros/jazzy/setup.bash && source /ros2_ws/install/setup.bash
```

`.env` : `SIM_SCENE` (vide = scène pick & place `scene_tek5.xml`, `scene_ouija.xml` =
phase 2) et `CAMERA_TF` (`sim` = pose caméra donnée par le simulateur, `calib` = estimée
par le nœud `camera_calib` à partir des marqueurs ArUco, comme sur le vrai robot, voir
« Calibration caméra »). Changer l'un ou l'autre : `docker compose restart control perception`.

Le code Python est monté en volume et installé en symlink : après une modification,
`docker compose restart control` (ou `perception`) suffit, pas de rebuild. Attention,
redémarrer `control` ferme RViz.

## Nodes

| Node | Package | Rôle |
|---|---|---|
| `so101_driver` | `so101_driver` | Seul à parler au backend (MuJoCo ou bras réel). Publie `/joint_states` (30 Hz, rad), écoute `/joint_command` (rad, pince 0-100 %). En simu : caméra `/external_cam/image_raw` + `camera_info` (10 Hz), TF statique `world -> external_cam_optical` (si `publish_camera_tf`, c.-à-d. `CAMERA_TF=sim`) et toujours la vérité terrain `world -> external_cam_optical_gt`, `/debug/*_gt`, service `/sim/randomize_object`. |
| `camera_calib` | `so101_perception` | Pose de la caméra (TF `world -> external_cam_optical`) estimée par solvePnP sur les marqueurs ArUco de la feuille A4 (`calib/`) ou de la planche Ouija, posées à un endroit connu du repère robot. Compare à la vérité terrain en simu, sauve dans `ros2_ws/calibration/camera_tf.json`. `/camera_calib/status`, `/camera_calib/debug_image`. |
| `so101_brain` | `so101_brain` | FK (pinocchio) sur `/end_effector_pose` + marqueurs RViz `/end_effector_markers` (sphère et texte FK sur la pince, trace du TCP, cible IK), IK (services), machine à états pick & place. |
| `so101_perception` | `so101_perception` | Segmentation HSV du cube rouge et de la boîte, pixel -> rayon (K) -> intersection avec le plan z=2.5 cm -> `/object_position`, `/drop_box_position` (frame `world`), image annotée `/perception/debug_image`, marqueur `/object_marker`. |
| `ouija_perception` | `so101_perception` | Phase 2. ArUco (4 marqueurs aux coins) + solvePnP -> pose de la planche `/ouija/board_pose` ; anneau bleu -> `/ouija/planchette_position` ; lettre sous l'anneau `/ouija/pointed` ; `/ouija/debug_image`, `/ouija/targets_markers` (RViz). |
| `so101_ouija` | `so101_brain` | Phase 2. Question -> réponse (Oracle à règles) -> déplace la planchette lettre par lettre puis sur FIN, en boucle fermée avec la caméra. Services `/ouija/ask`, `/ouija/spell`, `/ouija/stop`, topics `/ouija/question`, `/ouija/answer`, `/ouija/state`. |

## Services

```bash
# IK : amener la pince (approche verticale par défaut) à un point du repère world
ros2 service call /go_to_target so101_interfaces/srv/GoToTarget "{target_pose: {header: {frame_id: world}, pose: {position: {x: 0.25, y: 0.05, z: 0.08}}}}"
# joints en degrés (5) + pince en % (optionnel)
ros2 service call /set_joint_positions so101_interfaces/srv/SetJointPositions "{positions: [0, -90, -5, 90, 0, 100]}"
# ligne droite cartésienne (IK par points tous les 1 cm), orientation = idem go_to_target
ros2 service call /move_line so101_interfaces/srv/MoveLine "{target_pose: {header: {frame_id: world}, pose: {position: {x: 0.25, y: -0.05, z: 0.08}}}, speed: 0.05}"
# pick & place complet (bloquant, ~16 s), état courant sur /brain/state
ros2 service call /pick_and_place std_srvs/srv/Trigger
ros2 service call /brain/stop std_srvs/srv/Trigger          # arrêt d'urgence logiciel
ros2 service call /sim/randomize_object std_srvs/srv/Trigger # simu : déplacer le cube
```

Test automatique de robustesse (N essais aléatoires, vérifiés avec la vérité terrain) :

```bash
python3 /ros2_ws/tools/trial_pick_place.py 10
```

Deux autres outils de mise au point (simulation) : `tools/grasp_bench.py` rejoue toute la séquence
de prise hors ROS (physique MuJoCo + l'IK du brain, ~5 s par essai, perception parfaite ou erreur
simulée `OBJ_ERR=8,0`) sur une grille de positions, pour régler `grasp_offset` / `grasp_height` ;
`tools/perception_bias.py N` mesure l'erreur de détection du cube par rapport à la vérité terrain
sur N positions aléatoires (biais moyen, pire erreur).

Téléopération clavier : `python3 /ros2_ws/teleop_keyboard.py`.

### Vidéos de démo (sans écran)

```bash
cd docker
./run.sh video 3 main   # vue 3D MuJoCo plein cadre      -> ros2_ws/rec/pick_place_sim_main.mp4
./run.sh video 3 rviz   # vue RViz (TF + FK du TCP)      -> ros2_ws/rec/pick_place_rviz.mp4
./run.sh video 3 grid   # 2x2 : MuJoCo, RViz (TF + FK), caméra, détection, 1920x1080 -> ros2_ws/rec/pick_place_grid.mp4
                        # (vidéo de rendu : les trois vues exigées dans un seul cadre)
./run.sh video 3        # 3 panneaux (3D, caméra, perception) -> ros2_ws/rec/pick_place_sim.mp4
./run.sh video-ouija "Qui est le meilleur groupe ?" NOUS   # phase 2 -> ros2_ws/rec/ouija_demo.mp4
./run.sh video-reel "Qui est le meilleur groupe ?" NOUS    # phase 2 sur le VRAI robot -> ros2_ws/rec/ouija_reel.mp4
./run.sh video-reel "Es-tu vivant ?" LLM                   # idem, la question passe par le LLM (ollama)
# clip téléphone de la scène à côté de l'enregistrement (son du téléphone gardé, décalage en s optionnel) :
docker compose exec control bash /ros2_ws/tools/side_by_side.sh /ros2_ws/rec/tel.mp4 /ros2_ws/rec/ouija_reel.mp4 /ros2_ws/rec/rendu.mp4 [decalage]
```

`tools/record_demo.py` lance N essais aléatoires (`/sim/randomize_object` + `/pick_and_place`), vérifie
chaque dépôt avec la vérité terrain et écrit un MP4 H.264 1280x720 à 20 i/s. La vue RViz tourne
**dans le conteneur** sans écran : `rviz2 --fullscreen -d so101_video.rviz` sur un serveur X virtuel
(Xvfb, display :90 à :99), capturé par `ffmpeg -f x11grab`. L'encart en haut à gauche compare la
position du TCP calculée par notre FK (`/end_effector_pose`) et celle de l'arbre TF
(`robot_state_publisher`, même URDF, calcul indépendant) : écart 0,0 mm. Dans RViz (config
`so101.rviz` aussi) : `/end_effector_markers` (sphère jaune + texte FK sur la pince, trace bleue du
TCP, cible IK rose), `/debug/sim_markers` (cube rouge, boîte, planchette = vérité terrain de la
simulation, affichage seulement) et `/object_marker` (cube vert en fil de fer = détection).
Vérification manuelle de la FK : `ros2 run tf2_ros tf2_echo world gripper_frame_link` doit donner
la même position que `ros2 topic echo /end_effector_pose`.

`tools/record_ouija.py` (mode `video-ouija`) : vue 3D par-dessus l'épaule du robot (les lettres de la
planche se lisent de son côté), encart `/ouija/debug_image`, bandeau avec la question, les lettres
épelées (une par état `DWELL`) et la lettre vue sous l'anneau par la perception. La réponse est
donnée au script et épelée par `/ouija/spell` (pas de LLM), la vidéo est donc reproductible.

## Fréquences et QoS

| Topic | Fréquence | QoS |
|---|---|---|
| `/external_cam/image_raw` | 10 Hz en simu (`camera_fps`), `CAMERA_FPS` sur la vraie caméra | **Best Effort**, keep last 5 (profil « sensor data ») |
| `/external_cam/camera_info` | 10 Hz | Reliable (petit, doit arriver) |
| `/joint_states` | 30 Hz | Reliable, keep last 10 |
| `/joint_command` | 50 Hz pendant un mouvement | Reliable, keep last 10 |
| `/end_effector_pose` | 20 Hz | Reliable |
| `/object_position`, `/drop_box_position` | cadence image (10 Hz) | Reliable |
| `*/debug_image` | cadence image | Best Effort, keep last 2 |

Pourquoi Best Effort pour les images : une image perdue est remplacée par la suivante 100 ms
plus tard, alors qu'une image retransmise arrive en retard et ferait agir le bras sur une position
périmée. États et commandes restent Reliable : une commande perdue, c'est un pas manqué.
Compatibilité : un abonné Best Effort reçoit d'un éditeur Reliable ou Best Effort ; un abonné
Reliable ne reçoit **rien** d'un éditeur Best Effort (symptôme classique : « No Image » dans RViz,
il faut passer le display en Best Effort, c'est fait dans `so101.rviz`).

```bash
ros2 topic hz /external_cam/image_raw        # ~10 Hz
ros2 topic info -v /external_cam/image_raw   # QoS de l'éditeur et de chaque abonné
ros2 topic hz /joint_states                  # ~30 Hz
```

## Phase 2 : Ouija

Scène `sim/so101_sim/assets/so101/scene_ouija.xml` : planche 30 x 50 cm posée devant le
robot (centre x = 23,5 cm), texture et positions des lettres générées par
`ouija/make_board.py` (`ouija_board.png` + `ouija_layout.json`, à regénérer après toute
modification : `docker compose exec -T --user ubuntu control python3
/opt/so101/sim/so101_sim/assets/so101/ouija/make_board.py`). Les lettres sont sur des arcs
centrés sur la base du robot (A-M r = 29,5 cm, N-Z 25,5 cm, 0-9 21,5 cm, OUI/FIN/NON 18 cm :
portée mesurée de la pince verticale 13-34 cm). La planchette est un anneau bleu (r 2,1 cm,
trou 1,1 cm) : la pointe de la pince fermée descend dans le trou et le tire dans n'importe
quelle direction (un palet poussé par la pointe arrondie glissait de côté). La caméra est
au-dessus du bord éloigné de la planche, inclinée de 20° vers le robot.

**Impression du plateau (vrai robot)** : `python3 sim/so101_sim/assets/so101/ouija/make_board_print.py`
-> `ouija_board_A4.pdf`, 4 pages A4 à imprimer à 100 % (quarts de 15 x 25 cm, fond blanc),
à couper sur le cadre noir et assembler bord à bord sur un carton : le plateau fait
30 x 50 cm, exactement la géométrie de la simu (`ouija_layout.json`). La planchette est
un anneau bleu à fabriquer (impression 3D) : diamètre 42 mm, trou 22 mm, hauteur 15 mm.

```bash
# dans un shell ROS du conteneur control
ros2 service call /ouija/ask so101_interfaces/srv/Ask "{question: 'est-ce que tu es vivant ?'}"
ros2 service call /ouija/spell so101_interfaces/srv/Ask "{question: SALUT}"   # épelle tel quel
ros2 topic pub --once /ouija/question std_msgs/msg/String "{data: 'bonjour'}"  # sans attendre
ros2 topic echo /ouija/state          # OBSERVE / ALIGN / DESCEND / MOVE / RETREAT / DWELL X
ros2 topic echo /ouija/pointed        # lettre vue sous la planchette par la caméra
ros2 service call /ouija/stop std_srvs/srv/Trigger
ros2 service call /sim/reset_planchette std_srvs/srv/Trigger   # simu : anneau sur FIN
# vérifications
python3 /ros2_ws/tools/check_ouija_board.py --observe   # ArUco vs vérité terrain
python3 /ros2_ws/tools/ouija_check.py                   # planchette / planche vs vérité terrain
```

Image de debug dans RViz : topic `/ouija/debug_image` (type Image), marqueurs
`/ouija/targets_markers` et `/ouija/target_marker`.

Boucle par caractère : `OBSERVE` (bras replié hors champ, attente d'une détection fraîche)
-> `ALIGN` (au-dessus de l'anneau) -> `DESCEND` (pointe dans le trou, z = 1,2 cm au-dessus de
la planche) -> `MOVE` (ligne droite vers la lettre, service `/move_line`) -> `RETREAT` ->
`OBSERVE` : si l'anneau est à plus de 8 mm de la lettre, nouveau déplacement (3 max) ->
`DWELL` 1,5 s. Lettre doublée : petit écart de côté puis retour. Erreur mesurée en simu :
2 à 6 mm par lettre en un seul déplacement.

Les réponses viennent d'une petite IA locale et gratuite : le conteneur `ollama` sert le
modèle `qwen2.5:1.5b` (~1 Go, CPU, hors ligne). Première utilisation :
`docker compose up -d ollama && docker compose exec ollama ollama pull qwen2.5:1.5b`.
Le nœud ouija (`LlmOracle`) lui envoie la question avec une consigne « un mot de 8 lettres
max, ou un nombre, ou OUI/NON », nettoie la réponse (majuscules sans accent, 10 caractères
max) et, si le modèle ne répond pas, retombe sur les règles (`Oracle` : salutations, nom,
« combien » -> chiffre, sinon OUI/NON déterministe). Paramètres : `ai` (`llm` ou `rules`),
`ollama_url`, `ollama_model`, `max_answer_chars`, `llm_timeout`. Autre modèle :
`docker compose exec ollama ollama pull llama3.2:3b` puis `ollama_model`.

Paramètres du nœud ouija : `observe_joints_deg` [0, -100, 40, -80, 0], `safe_height` 0,08,
`finger_height` 0,012, `move_speed` 0,05 m/s, `position_tolerance` 0,008,
`max_moves_per_letter` 3, `dwell_time` 1,5.

## Calibration caméra (TF world -> caméra)

Le sujet : la vérité terrain `get_camera_extrinsics()` sert à débloquer, mais « sur le vrai
robot personne ne vous donne cette matrice ». Comme montré au kickoff (feuille A4 avec un
marqueur), le nœud `camera_calib` estime la pose de la caméra :

1. une feuille A4 avec 4 marqueurs ArUco 4x4_50 de 7 cm (ids 10-13) est posée à un endroit
   **mesuré** du repère robot : argument `sheet_pose` = "x y yaw°" du centre de la feuille
   (repère dessiné dessus : x = petit côté, y = grand côté ; z = 1 mm implicite). Sur le
   vrai robot c'est `CALIB_SHEET_POSE` dans `docker/.env` (idem `OUIJA_BOARD_POSE` pour la
   planche). En simu elle est dans `scene_tek5.xml` à (-0,15, 0,20), derrière-gauche du
   bras, hors zone du cube ;
2. solvePnP sur les 16 coins (raffinement sous-pixel AprilTag) -> `T_camera_world`,
   moyenne de 30 images -> TF statique `world -> external_cam_optical` ;
3. en simu le driver publie toujours la vérité terrain sous `external_cam_optical_gt` : le
   nœud logge son erreur. Mesuré : 1,4 mm / 0,05° (scène phase 1, rendu sans ombres, le
   défaut depuis le 17 novembre ; 5 mm / 0,23° avec les ombres, qui dégradent les coins),
   6,8 mm / 0,40° sur la scène Ouija calibrée avec les marqueurs de la planche. Pick & place
   3/3, Ouija 1-2 mm par lettre avec la TF calibrée ;
4. le résultat est sauvé (`ros2_ws/calibration/camera_tf.json`) : `calib_mode:=load`
   (`CALIB_MODE=load` dans `.env`) republie la TF sans la feuille, à utiliser quand on l'a
   enlevée du plan de travail. Attention, en simu chaque session réécrit ce fichier.

```bash
# feuille à imprimer à 100 % (pdf A4) + layout
python3 sim/so101_sim/assets/so101/calib/make_calib_sheet.py   # calib_sheet.pdf/.png/.json
# .env : CAMERA_TF=calib puis
docker compose restart control perception
docker compose logs -f perception     # "vs ground truth: position error ... mm"
# image de debug : RViz, Image sur /camera_calib/debug_image (marqueurs + repère robot dessiné)
ros2 topic echo /camera_calib/status  # "sampling k/30" puis "done"
```

Paramètres (`perception.launch.py`) : `camera_tf` (sim | calib), `calib_mode` (calibrate |
load), `calib_sheets` (layouts json), `sheet_pose` et `board_pose` ("x y yaw°"),
`params_file` (yaml de seuils HSV pour la vraie caméra, `ros2_ws/config/perception_real.yaml`,
réglé avec `./run.sh hsv`). Nœud : `num_samples` 30, `min_markers` 3 (avec 2 marqueurs
l'erreur triple), `calib_file`.

Sur le vrai robot, la séquence complète (intrinsèques au damier, feuille, calibration,
vérification) est dans `docs/GUIDE_ROBOT_REEL.md`.

## Machine à états du brain

`OBSERVE` (bras relevé hors du champ caméra, pince ouverte, attente de détections
fraîches) -> `APPROACH` (12 cm au-dessus, re-planifie si le cube bouge de plus de 2 cm)
-> `DESCEND` (ligne droite, TCP à z = 3 cm, `grasp_height`) -> `GRASP` -> `LIFT` -> contrôle de prise (cube toujours vu sur
la table = raté, nouvel essai, 2 max) -> `TRANSPORT` -> `RELEASE` dans la boîte ->
`RETREAT` -> `OBSERVE`. Le brain n'attend jamais la perception : il utilise la dernière
valeur connue avec une limite d'âge (`max_target_age`, 2 s).

Résultats mesurés en simu : IK < 1 mm, perception < 2 mm (cube et boîte),
pick & place 10/10 sur positions aléatoires.

Point de prise : le repère outil `gripper_frame_link` (le TCP de la FK/IK) est la face
interne de la mâchoire FIXE, pas le milieu des mors : le doigt fixe occupe x_tcp ∈ [0, +14] mm,
la mâchoire mobile se ferme depuis le côté −x_tcp (mesuré sur le modèle). Le cube de 5 cm est
creux, avec un trou en losange de 38 mm dans chaque face : le brain place le TCP à
`grasp_offset` = −7 mm du centre le long de +x_tcp, de sorte que le doigt fixe soit centré dans le
trou du dessus et descende à l'intérieur du cube (z = 3 cm). La mâchoire mobile pousse alors le
cube jusqu'à ce que sa paroi opposée bute sur le doigt, qui s'engage dans le trou latéral : le cube
est accroché et ne peut plus glisser (tolérance ≈ ±12 mm, le bord incliné du trou recentre le
cube). Une simple pince sur les faces externes ne tient pas avec cette pince : la mâchoire mobile
est un levier qui appuie sur l'arête haute vers le bas, et le cube est expulsé au levage. Pour un
objet plein, mettre `grasp_offset` = +(demi-largeur + 3 mm) : le TCP juste au-delà de la face
proche. La descente et le levage sont des lignes droites cartésiennes (sinon le trajet articulaire
courbe fait accrocher le cube). Côté simulation, `scene_tek5.xml` active `impratio=10` et
`noslip_iterations=5` : sans cela le frottement « mou » de MuJoCo laisse fluer le cube serré hors
des mors (recommandation MuJoCo pour la manipulation). La perception ignore un blob coupé par le bord
de l'image (centroïde faussé, avertissement « object partly out of the camera view ») : la caméra doit
voir toute la zone de travail (en sim, `fovy=70`). Pose d'observation (`observe_joints_deg`) :
bras levé, pince horizontale à 30 cm de haut et 9 cm devant la base, pour ne balayer aucun cube au départ.

Orientation dans `GoToTarget` / `MoveLine` : quaternion identité = pince verticale, roulis
libre ; sinon l'axe z du quaternion = direction d'approche et l'axe x = direction des
mâchoires (obtenue avec `wrist_roll`, libre quand la pince pointe vers le bas).

## Paramètres utiles (brain)

`observe_joints_deg`, `approach_height`, `grasp_height`, `lift_height`, `drop_height`,
`max_target_age`, `max_grasp_retries`, `replan_distance`, `motion_tolerance`, `grasp_offset`
(−7 mm, voir « Point de prise »), `grasp_min_gripper_pct` (1 % : cube accroché, la pince se ferme encore à
2-8 % ; le contrôle principal est la perception).

Correction en boucle fermée de `go_to` (fléchissement du vrai bras sous gravité) : après le
mouvement, le brain mesure l'erreur FK (positions moteurs lues) ; si elle dépasse
`goto_correction_min_error` (3 mm), il décale la cible de `goto_correction_gain` (0,8) fois
l'erreur, re-résout l'IK depuis la posture courante et recommence, `goto_corrections` (2)
fois au plus. Si l'erreur empire, il revient à la meilleure commande. Actif pour
`/go_to_target` et la descente sur le cube, pas pour les transits. Mettre `goto_corrections`
à 0 pour le couper. Pour le tester en simu (le bras simulé ne fléchit pas, erreur < 1 mm) :
`ros2 param set /so101_brain goto_correction_min_error 0.0002`.

## Bras réel

`.env` : `USE_SIM=false`, `REAL_ROBOT_PORT` (`ls /dev/ttyACM*`, adaptateur série QinHeng
CH343), `CALIBRATION_FILE` (un des fichiers de `calibration/`, celui qui correspond à ce qui
est écrit dans les moteurs : `tools/read_motor_calibration.py` le vérifie), puis
`docker compose up -d control viz`. Le driver refuse de démarrer si la calibration du
fichier ne correspond pas à celle écrite dans les moteurs.

Outils du bras réel (`ros2_ws/tools/`, détails dans `docs/GUIDE_ROBOT_REEL.md`) :
- `read_motor_calibration.py` : lit la calibration stockée dans les moteurs (lecture seule), la
  compare au fichier du driver et la sauve dans `ros2_ws/calibration/` ;
- `write_motor_calibration.py` : réécrit le fichier du driver dans les moteurs (quand ils ont
  perdu leur calibration mais que le fichier est bon) ;
- `recalibrate_arm.py` : recalibration complète, interactive (même procédure que `lerobot-calibrate`) ;
- `monitor_positions.py` : positions brutes des 6 moteurs en direct, driver arrêté ;
- `touch_table.py` : hauteur de la table dans le repère robot, pince verticale (-> `table` de
  `ros2_ws/calibration/workspace.yaml`) ;
- `compare_touch.py` : écart entre le cube vu par la caméra et la pointe de la pince posée dessus
  (-> `xy_offset` de `workspace.yaml`) ;
- `hover_object.py` : met la pince au-dessus du cube vu par la perception (test avant une prise) ;
- `hsv_tune.py` (`./run.sh hsv`) : seuils de couleur sous la vraie lumière -> `ros2_ws/config/perception_real.yaml` ;
- `wait_topic.py` : utilisé par `run.sh` pour attendre que la perception publie.

Comportement du driver sur le vrai bras (LeRobot `SO101Follower`) :
- à la connexion le couple est **activé** : le bras tient sa position et ne bouge que sur
  `/joint_command` (rien n'est envoyé tant qu'aucune commande n'arrive) ;
- à l'arrêt du nœud (`docker compose stop/restart control`, Ctrl-C) le couple est
  **coupé** : le bras devient mou et retombe. Toujours le ramener en position de repos
  avant d'arrêter ;
- `ARM_TORQUE=false` : moteurs mous, lecture seule de `/joint_states` (premier contact,
  vérifier le sens des axes, relever la position de repos à la main) ;
- `ARM_MAX_STEP=0.1` : chaque commande est plafonnée à ±0,1 rad autour de la position
  lue (articulations du bras, pas la pince). Les trajectoires du brain (pas de 50 Hz) ne
  sont pas affectées, une commande brute aberrante (touche `o` de la téléop, service
  avec un gros saut) est réduite à un petit pas. `0` = désactivé.

Premier contact, dans l'ordre :

```bash
# 1. .env : USE_SIM=false, ARM_TORQUE=false, ARM_MAX_STEP=0.1 ; brancher l'alimentation 12 V
ls -la /dev/ttyACM0
docker compose up -d control viz
docker compose logs -f control        # "Robot connected: SO101Follower", "Torque disabled"
docker compose exec control bash
source /opt/ros/jazzy/setup.bash && source /ros2_ws/install/setup.bash
ros2 topic echo /joint_states         # bouger chaque axe à la main : sens et zéro vs RViz
# 2. mettre le bras à la main en position de repos (replié), noter /joint_states = pose de repos
# 3. .env : ARM_TORQUE=true ; docker compose restart control  (le bras est au repos, il ne tombe pas)
python3 /ros2_ws/teleop_keyboard.py   # petits pas (touche x), un axe à la fois
ros2 service call /set_joint_positions so101_interfaces/srv/SetJointPositions "{positions: [0, -100, 40, -80, 0, 0]}"  # pose d'observation
ros2 service call /go_to_target so101_interfaces/srv/GoToTarget "{target_pose: {header: {frame_id: world}, pose: {position: {x: 0.25, y: 0.0, z: 0.10}}}}"
# 4. avant d'arrêter : pose de repos, puis docker compose stop control
```

### Premier contact réel (29 octobre 2026) : résultats

- Le fichier de calibration du 9 octobre ne correspondait pas aux moteurs (tous les offsets
  différents). La calibration stockée dans les moteurs a été relue avec LeRobot
  (`bus.read_calibration()`) et sauvée dans `calibration/so101_follower_from_motors_2026-10-29.json`,
  qui est maintenant `CALIBRATION_FILE`. Même convention que le modèle : la pose
  d'observation [0, -100, 40, -80, 0] donne le même bras que la simu (pince en haut,
  derrière la base).
- Alimentation : une première alim donnait 20 V (servos en « Input voltage error », limite
  14 V, ils refusent de bouger). Avec la bonne alim : 9 V (12 V = couple nominal).
- Pose de repos repliée lue à la main : pan 0, lift -105, elbow 99, wrist_flex 48,
  roll -87 (deg).
- Mouvements : `/set_joint_positions` suit à 1° près. `/go_to_target` (0,25, 0, 0,12) :
  erreur 8,8 mm et 12,7° sur l'axe d'approche, surtout du fléchissement sous gravité
  (gain P des servos mis à 16 par LeRobot, 9 V). Le plafond `ARM_MAX_STEP=0.1` s'est
  déclenché sur `shoulder_lift` pendant ce mouvement (le servo prend du retard sur la
  trajectoire) : il ralentit sans bloquer.
- Pour lire le bus en dehors du driver (tension, calibration) : arrêter le driver d'abord,
  un seul programme peut ouvrir `/dev/ttyACM0`.

### Caméra réelle

Le driver ne publie que la caméra simulée. En vrai, le service `camera` (profil `real`,
jamais lancé par défaut) publie une caméra USB UVC sur les mêmes topics
`/external_cam/image_raw` + `camera_info` (nœud `v4l2_camera`), donc la perception ne
voit pas la différence. Testé avec la webcam du PC.

```bash
# .env : CAMERA_DEVICE (/dev/video0), CAMERA_WIDTH/HEIGHT/FPS, CAMERA_PIXEL_FORMAT (YUYV,
#        GREY pour une caméra monochrome), CAMERA_ENCODING (rgb8, mono8)
docker compose build camera                       # une fois
docker compose up -d camera
docker compose exec camera bash -lc "source /opt/ros/jazzy/setup.bash && ros2 topic hz /external_cam/image_raw"
docker compose exec camera v4l2-ctl --list-formats-ext   # formats / tailles / fps de la caméra
docker compose exec camera v4l2-ctl -c focus_automatic_continuous=0   # autofocus OFF si la caméra en a un
```

Intrinsèques (K + distorsion), une fois par caméra, avec un damier imprimé (8x6 coins
intérieurs, cases de 25 mm) ; bouton CALIBRATE puis COMMIT : le fichier
`ros2_ws/calibration/camera_info.yaml` est écrit et publié dans `camera_info` :

```bash
xhost +local:docker
docker compose exec camera bash -lc "source /opt/ros/jazzy/setup.bash && ros2 run camera_calibration cameracalibrator --size 8x6 --square 0.025 --no-service-check --ros-args -r image:=/external_cam/image_raw -r camera:=/external_cam"
docker compose restart camera
```

Sans ce fichier, `camera_info` contient K = 0 : les nœuds de perception l'ignorent et
préviennent (« camera_info without intrinsics »). Ensuite la pose de la caméra : feuille
ArUco + `CAMERA_TF=calib` (section « Calibration caméra »).

Choix de la caméra : la perception phase 1 (cube rouge) et la planchette Ouija (anneau
bleu) sont segmentées par **couleur** (HSV). Une caméra **monochrome** (OV9281 « mono »,
global shutter) est parfaite pour les ArUco mais casse ces deux détections : il faudrait
alors coller des ArUco sur le cube et la planchette (ou segmenter par forme). Une webcam
couleur UVC classique (autofocus désactivable) suffit ; une caméra de profondeur
(RealSense, OAK-D) marche aussi via son flux couleur (nœud `realsense2_camera`, déjà
calibré), mais la profondeur n'est pas utilisée : les objets sont sur un plan connu.
