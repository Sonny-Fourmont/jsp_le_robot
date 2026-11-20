# Guide : passer du simulateur au vrai robot (pick & place)

Ce guide décrit, dans l'ordre, ce qu'il faut faire pour que le pick & place du cube rouge vers la boîte fonctionne sur le vrai SO-ARM101 avec une webcam. Il complète `ros2_ws/README_pipeline.md`, qui reste la référence détaillée.

Toutes les commandes `docker compose` se lancent depuis le dossier `docker/`.

Voir aussi `vrai_robot_jourJ.md` : la même séquence condensée, avec le lanceur `docker/run.sh`
(`./run.sh real`, `./run.sh hsv`, `./run.sh video-reel`) et la phase 2 (planche Ouija).

## 0. Matériel nécessaire

| Élément | Remarque |
|---|---|
| Bras SO-ARM101 + alimentation 12 V | Une alim à 20 V met les servos en erreur (limite 14 V). |
| Câble USB du bras | Adaptateur série QinHeng CH343, en général `/dev/ttyACM0`. |
| Cube de 5 cm avec trous, **rouge** | La détection se fait par couleur (HSV). |
| Boîte ou corbeille **brun/orange** | Plage `box_hsv_*` de `perception_node.py`. Autre couleur : ajuster les seuils. |
| Webcam USB couleur | Autofocus désactivable. Pas de caméra monochrome. |
| Damier imprimé, 8x6 coins intérieurs, cases de 25 mm | Pour les intrinsèques de la webcam. Fichier : `sim/so101_sim/assets/so101/calib/checkerboard_8x6_25mm.pdf`, à imprimer à 100 % (barre de contrôle de 10 cm). |
| Feuille A4 avec marqueurs ArUco | Pour la pose de la caméra (étape 4). |
| Support fixe pour la webcam | Elle ne doit plus bouger après la calibration. |

## 1. Configuration (`docker/.env`)

Valeurs à avoir pour le réel :

```
USE_SIM=false
CAMERA_TF=calib
REAL_ROBOT_PORT=/dev/ttyACM0        # à vérifier (étape 2)
CALIBRATION_FILE=<chemin absolu du repo>/calibration/so101_follower_from_motors_2026-10-29.json
ARM_TORQUE=true
ARM_MAX_STEP=0.1                    # plafond de sécurité par commande (rad)
ARM_MAX_SPEED=0.3                   # vitesse max (rad/s), à monter après les premiers essais
CAMERA_DEVICE=/dev/video0           # à vérifier (étape 2)
CAMERA_WIDTH=640
CAMERA_HEIGHT=480
CAMERA_FPS=10
CAMERA_PIXEL_FORMAT=YUYV
CAMERA_ENCODING=rgb8
```

Le fichier de calibration du 9 octobre (`my_awesome_follower_arm.json`) ne correspond pas aux moteurs : le driver refuse de démarrer avec lui. Utilisez celui du 29 octobre.

## 2. Branchements

1. Branchez l'alimentation du bras et son câble USB, puis la webcam.
2. Vérifiez que le système les voit :

```bash
ls /dev/ttyACM* /dev/video*
v4l2-ctl --list-devices
```

3. Si les ports diffèrent, corrigez `REAL_ROBOT_PORT` et `CAMERA_DEVICE` dans `.env`.

## 3. Démarrer le bras seul

**Avant de lancer** : mettez le bras à la main en position repliée. À la connexion, il garde sa position.

```bash
cd docker
xhost +local:docker
docker compose up -d control viz
docker compose logs -f control      # attendre "Robot connected: SO101Follower"
```

Si le driver refuse de démarrer, la calibration du fichier ne correspond pas à celle écrite dans les moteurs.

Test d'un petit mouvement vers la pose d'observation :

```bash
docker compose exec control bash
source /opt/ros/jazzy/setup.bash && source /ros2_ws/install/setup.bash
ros2 topic echo /joint_states --once
ros2 service call /set_joint_positions so101_interfaces/srv/SetJointPositions "{positions: [0, -100, 40, -80, 0, 0]}"
```

RViz (autre terminal) :

```bash
docker compose exec control bash -lc "source /ros2_ws/install/setup.bash && rviz2 -d /ros2_ws/so101.rviz"
```

Dans RViz : ajoutez les displays Marker (`/object_marker`, `/drop_box_marker`) et Image (`/perception/debug_image`, **Reliability Policy = Best Effort**).

**Avant tout arrêt du driver** (`docker compose stop control` ou `restart`) : ramenez le bras en position de repos. À l'arrêt le couple est coupé et le bras retombe.

Un seul programme à la fois peut ouvrir `/dev/ttyACM0`.

## 4. Webcam : intrinsèques

```bash
docker compose build camera                 # une seule fois
docker compose up -d camera
docker compose exec camera v4l2-ctl -c focus_automatic_continuous=0    # autofocus OFF
docker compose exec camera bash -lc "source /opt/ros/jazzy/setup.bash && ros2 topic hz /external_cam/image_raw"
```

Calibration avec le damier (`sim/so101_sim/assets/so101/calib/checkerboard_8x6_25mm.pdf`, imprimé à 100 %, collé sur un support plat et rigide) :

```bash
xhost +local:docker
docker compose exec camera bash -lc "source /opt/ros/jazzy/setup.bash && ros2 run camera_calibration cameracalibrator --size 8x6 --square 0.025 --no-service-check --ros-args -r image:=/external_cam/image_raw -r camera:=/external_cam"
```

Bougez le damier dans toute l'image (angles, distances, inclinaisons), puis cliquez **CALIBRATE** et **COMMIT**. Le fichier `ros2_ws/calibration/camera_info.yaml` est écrit.

```bash
docker compose restart camera
```

Sans ce fichier, K vaut 0 et la perception ignore les images.

## 5. Webcam : pose par rapport au robot

1. Fixez la webcam de façon rigide, avec la zone de travail dans le champ.
2. Générez la feuille et imprimez-la à **100 %** :

```bash
python3 sim/so101_sim/assets/so101/calib/make_calib_sheet.py     # calib_sheet.pdf / .png / .json
```

3. Posez la feuille dans le champ de la caméra, **sur la table**, un coin contre le socle du robot. Elle ne doit plus bouger pendant la calibration.
4. **Mesurez** la position du centre de la feuille par rapport à la base du robot : x, y, z (m) et angle de rotation (yaw, degrés). Repère dessiné sur la feuille : x = petit côté, y = grand côté.
5. Mettez ces valeurs dans `calib_poses` (`ros2_ws/src/so101_perception/launch/perception.launch.py`). Les valeurs actuelles sont celles de la simulation. Format : 4 valeurs par feuille, `[x, y, z, yaw]`. Les 4 premières concernent la feuille A4, les 4 suivantes la planche Ouija.
6. Lancez la calibration :

```bash
docker compose up -d perception
docker compose logs -f perception        # "sampling k/30" puis "done"
docker compose exec perception bash -lc "source /opt/ros/jazzy/setup.bash && ros2 topic echo /camera_calib/status"
```

7. Le résultat est sauvé dans `ros2_ws/calibration/camera_tf.json`. Retirez ensuite la feuille de la zone de travail et passez en mode `load` (argument `calib_mode:=load` du launch) pour republier la TF sans la feuille.

Note : le `camera_tf.json` présent dans le repo date de la simulation. Il sera écrasé par votre calibration réelle.

## 6. Tester la perception

1. Posez le cube rouge à une position mesurée à la règle.
2. Comparez avec `/object_position` :

```bash
ros2 topic echo /object_position --once
```

L'écart doit rester de l'ordre du centimètre. Dans RViz, le marqueur vert doit se superposer à la position réelle.

3. Si le cube n'est pas détecté, ajustez les seuils HSV à votre éclairage (`object_hsv_*` dans `perception_node.py`).
4. Contrôlez `object_plane_z` (0,025 m, hauteur du centre du cube) : la table réelle doit être au même niveau que le zéro du robot.

## 7. Pick & place

Posez la boîte à portée du bras, gardez la main près de l'alimentation, puis :

```bash
ros2 service call /pick_and_place std_srvs/srv/Trigger
ros2 topic echo /brain/state            # dans un autre terminal
ros2 service call /brain/stop std_srvs/srv/Trigger     # arrêt d'urgence logiciel
```

États attendus : OBSERVE, APPROACH, DESCEND, GRASP, LIFT, TRANSPORT, RELEASE, RETREAT.

Une fois plusieurs essais concluants, montez `ARM_MAX_SPEED` (par exemple 0.6) et relancez `docker compose restart control` (après avoir remis le bras au repos).

## 8. Problèmes connus et dépannage

| Symptôme | Cause probable |
|---|---|
| Le driver refuse de démarrer | Fichier de calibration différent de celui des moteurs. |
| Servos en « Input voltage error », ils ne bougent pas | Mauvaise alimentation (limite 14 V). |
| `/go_to_target` s'arrête environ 9 mm trop bas | Fléchissement sous gravité (gain P des servos à 16 par LeRobot). Peut gêner la prise. |
| Le plafond `ARM_MAX_STEP` ralentit le mouvement | Normal : le servo prend du retard sur la trajectoire, la commande est réduite sans blocage. |
| La perception ne publie rien | `camera_info` sans intrinsèques (K = 0), ou TF de la caméra pas encore reçue : `docker compose logs perception`. |
| Image de debug vide dans RViz | Reliability Policy du display Image à passer en Best Effort. |
| Cube non détecté | Seuils HSV à ajuster à l'éclairage, ou cube pas assez rouge. |
| Le cube vert bouge pendant le transport | Normal : la position est calculée en supposant le cube posé sur la table. |
| Le port série est occupé | Un seul programme peut ouvrir `/dev/ttyACM0` : arrêtez le driver avant de lire le bus autrement. |

## 8 bis. Pièges rencontrés

- **Modifier `docker/.env` ne suffit pas avec `docker compose restart`** : le conteneur garde ses anciennes variables et ses anciens volumes. Utiliser `docker compose up -d control viz camera` (ou `--force-recreate`). RViz se ferme alors, il faut le relancer.
- **Les numéros de port changent à chaque rebranchement** (bras : `/dev/ttyACM0` ou `1`, webcam : `/dev/videoN`). Avant de relancer : `ls /dev/ttyACM*` et `for d in /sys/class/video4linux/video*; do echo "$(basename $d): $(cat $d/name) idx=$(cat $d/index)"; done` (prendre le device d'index 0), puis corriger `REAL_ROBOT_PORT` et `CAMERA_DEVICE`.
- **Le fichier de calibration des moteurs peut changer** (bras recalibré) : si le driver répond « does not match the motors », relire les moteurs avec `python3 /ros2_ws/tools/read_motor_calibration.py` (dans le conteneur `control`), copier le fichier produit dans `calibration/` et le mettre dans `CALIBRATION_FILE`.
- **Calibrateur de la webcam** : le bouton COMMIT fige la fenêtre (le nœud `v4l2_camera` ne répond pas au service) et n'écrit rien. Utiliser **SAVE**, puis :
  ```bash
  docker compose exec camera bash -lc "mkdir -p /tmp/calib && tar xzf /tmp/calibrationdata.tar.gz -C /tmp/calib"
  docker compose cp camera:/tmp/calib/ost.yaml ../ros2_ws/calibration/camera_info.yaml
  ```
  Puis mettre `camera_name: ugreen_camera:_ugreen_camera` dans ce fichier (sinon avertissement de nom) et `docker compose restart camera`. Ne pas redémarrer la caméra avant d'avoir copié le fichier : le redémarrage tue le calibrateur.
- **Le damier doit être un damier simple** (pas une planche ChArUco) : `--size 8x6 --square 0.025`.
- **Pose de repos par commande** : avec le couple actif, le bras est raide. `/set_joint_positions` avec `[0, -105, 99, 48, -87, 0]` le replie.

## 9. Livrables du projet

- Vidéo du robot réel qui attrape le cube, avec l'équipe visible dans la vidéo.
- Historique Git avec la contribution de chaque membre.
- Chaque membre doit comprendre toute la chaîne : caméra, calibration, perception, IK, contrôle, ROS2.
