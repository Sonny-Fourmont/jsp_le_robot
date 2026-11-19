# Le projet expliqué simplement, point par point

## 0. L'histoire en une phrase
Un bras robot a un œil, une caméra. Il voit un cube rouge sur la table, il le prend avec sa pince et le pose dans une boîte. On peut faire ça dans un **jeu vidéo** (la simulation MuJoCo) ou avec le **vrai bras**. Le programme est le même ; seul un interrupteur change : `use_sim`.

---

## 1. La présentation : calibration et perception

### La calibration caméra = apprendre à l'œil où il se trouve
La caméra voit des **pixels**, comme une photo. Le bras, lui, pense en **centimètres**, en partant de son pied. Pour traduire l'un en l'autre, il faut savoir deux choses sur la caméra :

1. **Comment est faite sa « lunette »** (les intrinsèques, notées K) : à quel point elle zoome, où est le centre de l'image, si elle déforme les bords.
   - En simulation, on nous donne ces valeurs.
   - Avec une vraie caméra, on lui montre un damier imprimé sous plein d'angles. Elle en déduit sa propre déformation.
2. **Où elle est posée et dans quelle direction elle regarde** (les extrinsèques).
   - On pose une feuille A4 avec 4 dessins carrés, les **marqueurs ArUco** (comme des QR codes), **à un endroit qu'on a mesuré**.
   - La caméra voit les coins des marqueurs. Une fonction, **solvePnP**, calcule alors : « pour voir la feuille comme ça, je dois être ici et regarder par là ».
   - On répète sur 30 images et on fait la moyenne. On obtient une erreur d'environ 5 mm par rapport à la vraie position.

### La perception = trouver le cube et dire où il est
1. **Trouver le rouge** : on garde seulement les pixels très rouges. C'est le seuil **HSV**, un tri par couleur. La boîte, elle, est trouvée grâce à sa propre couleur.
2. **Trouver le centre** de la tache rouge.
3. **Tirer un rayon laser imaginaire** depuis la caméra, à travers ce pixel.
4. **Regarder où le rayon touche** une hauteur connue. Le cube fait 5 cm et il est posé sur la table, donc son centre est à 2,5 cm de haut. Le point où le rayon atteint cette hauteur, c'est le cube.
5. On annonce la réponse sur `/object_position`, et pareil pour la boîte sur `/drop_box_position`.

> Pourquoi le rayon ? Une seule caméra ne sait pas à quelle distance sont les choses. Comme on sait que le cube est **sur la table**, on peut quand même calculer sa position.

---

## 2. L'architecture ROS2 = une équipe de petits robots qui se parlent

ROS2 sert à faire discuter plusieurs programmes, les **nodes**. Ils se parlent par des **topics**, comme des talkies-walkies : un node parle sur un canal et ceux qui écoutent ce canal reçoivent le message.

| Node | Son rôle (version enfant) | Il parle sur | Il écoute |
|---|---|---|---|
| **driver** | Les **muscles et l'œil**. C'est le seul à toucher le bras, vrai ou simulé. | `/joint_states` (« voilà mes angles », 30 fois par seconde), `/external_cam/image_raw` (la photo) | `/joint_command` (« bouge comme ça ») |
| **perception** | Les **yeux intelligents** | `/object_position`, `/drop_box_position` | la photo de la caméra |
| **camera_calib** | Celui qui **mesure où est la caméra** | la position de la caméra (une TF) | la photo |
| **brain** | Le **cerveau** : il calcule et décide | `/joint_command`, `/end_effector_pose`, `/brain/state` | les angles et les positions du cube et de la boîte |
| **robot_state_publisher** | Le **dessinateur** pour RViz | la forme du robot en 3D | les angles |

Chaque groupe vit dans sa **boîte Docker** : `control` (driver + brain), `perception`, `viz`. Toutes les boîtes sont sur le même réseau.

**À taper en démo :**
```bash
ros2 node list     # qui est dans l'équipe ?
ros2 topic list    # quels canaux de talkie-walkie existent ?
```

---

## 3. RViz et faire bouger le bras

RViz, c'est **la télé** qui montre le robot en 3D. On envoie un ordre depuis le terminal et le bras bouge, à l'écran comme en vrai :

```bash
ros2 service call /set_joint_positions so101_interfaces/srv/SetJointPositions "{positions: [0, -90, -5, 90, 0, 100]}"
```
Les 5 premiers nombres sont les angles des articulations en degrés. Le dernier est l'ouverture de la pince : 100 % = grande ouverte.

---

## 4. La cinématique directe (FK) = « mes angles → où est ma main ? »
Imagine ton bras : si tu sais l'angle de ton épaule, de ton coude et de ton poignet, tu peux deviner où est ta main.

**Comment on calcule.** Le robot est une chaîne de 5 articulations. Chaque articulation, c'est une rotation
d'un angle `q_i` (lu sur le moteur) suivie d'un décalage fixe jusqu'à l'articulation suivante (la longueur de la
pièce imprimée, écrite dans l'URDF). On empile les 5 transformations :

```
T_world->TCP = T_base · R(q1) · T_epaule · R(q2) · T_bras · R(q3) · T_avant-bras · R(q4) · T_poignet · R(q5) · T_pince
```

Le résultat est une matrice 4x4 : sa dernière colonne est la position du TCP (x, y, z) et ses 3 premières colonnes
son orientation. Le TCP (« tool center point »), c'est le repère `gripper_frame_link` de l'URDF : la face interne du
doigt fixe de la pince, c'est lui qu'on amène sur le cube.

- Le brain lit les angles sur `/joint_states` et fait ce calcul avec la bibliothèque **pinocchio** chargée avec l'URDF.
- Il publie le résultat sur `/end_effector_pose`, 20 fois par seconde, et des marqueurs RViz sur `/end_effector_markers` :
  une **sphère jaune** à la position calculée, le **texte** `FK:x=…,y=…,z=…` à côté de la pince, la **trace bleue** du TCP
  sur les 20 dernières secondes et une sphère rose à la cible IK courante.

**La preuve que la FK est juste, à montrer dans RViz.** RViz affiche le repère `gripper_frame_link` (trièdre rouge-vert-bleu)
grâce à **TF**, calculé par `robot_state_publisher` à partir du même URDF et des mêmes `/joint_states`, mais sans notre code.
Deux calculs indépendants de la même chaîne : la sphère jaune de notre FK tombe exactement sur l'origine du trièdre TF, et les
chiffres coïncident au dixième de millimètre. En direct, dans un shell ROS :

```bash
ros2 topic echo --once /end_effector_pose          # notre FK (pinocchio)
ros2 run tf2_ros tf2_echo world gripper_frame_link  # la même position vue par TF (robot_state_publisher)
ros2 service call /set_joint_positions so101_interfaces/srv/SetJointPositions "{positions: [0, -90, -5, 90, 0, 100]}"
```
Changer les angles, relire les deux : ils bougent ensemble. Dans RViz cocher les displays **TF**, **FK du TCP** et
**RobotModel**, puis bouger le bras avec `/set_joint_positions` ou `/go_to_target`.

La vidéo `ros2_ws/rec/pick_place_rviz.mp4` montre tout ça pendant un pick & place : l'encart en haut à gauche
compare en continu la position FK et la position TF (écart 0,0 mm), le cube rouge et la boîte sont la vérité terrain
de la simulation (`/debug/sim_markers`), le cube vert en fil de fer est ce que la perception a détecté.

---

## 5. La cinématique inverse (IK) = « je veux ma main ici → quels angles ? »
C'est le problème inverse, et c'est plus dur. Le brain fait comme toi quand tu cherches un objet à tâtons :
1. Il regarde où est la pince maintenant, avec la FK.
2. Il calcule de combien elle est loin de la cible.
3. Il bouge un tout petit peu les angles dans la bonne direction. C'est la **jacobienne** qui dit « si je tourne ce moteur, la main va par là ».
4. Il recommence jusqu'à être à **moins d'1 mm**.

Il essaie plusieurs positions de départ et garde la meilleure. Le bras n'a que 5 moteurs, donc il ne peut pas tout faire : la **position est prioritaire**, et garder la pince bien verticale passe en second.

```bash
ros2 service call /go_to_target so101_interfaces/srv/GoToTarget "{target_pose: {header: {frame_id: world}, pose: {position: {x: 0.25, y: 0.05, z: 0.08}}}}"
```
Traduction : « va à 25 cm devant toi, 5 cm sur le côté, 8 cm de haut ». Le sujet demande moins d'1 cm d'erreur.

---

## 6. La perception en direct
- Dans RViz, ajoute une Image sur `/perception/debug_image` : on voit le cube entouré.
- Ensuite :
```bash
ros2 topic echo /object_position              # la position 3D du cube
ros2 service call /sim/randomize_object std_srvs/srv/Trigger   # on déplace le cube…
```
Les nombres changent tout seuls : la caméra a suivi le cube.

---

## 7. Le pick & place = une recette en étapes
Le brain suit une **recette**, la machine à états. Il fait une étape à la fois :

1. **OBSERVE** : je lève le bras pour ne pas cacher le cube à la caméra et j'attends de le voir.
2. **APPROACH** : je vais 12 cm au-dessus du cube.
3. **DESCEND** : je descends jusqu'au cube.
4. **GRASP** : je ferme la pince.
5. **LIFT** : je soulève. Si la caméra voit encore le cube sur la table, c'est raté et je réessaie, 2 fois au maximum.
6. **TRANSPORT** : je vais au-dessus de la boîte.
7. **RELEASE** : j'ouvre la pince et le cube tombe dedans.
8. **RETREAT** : je m'écarte, puis je retourne à OBSERVE.

**Pourquoi c'est robuste :**
- Le cerveau **n'attend jamais** les yeux. Il utilise la dernière position connue, à condition qu'elle ait moins de 2 secondes.
- Si quelqu'un **déplace le cube** de plus de 2 cm pendant l'approche, il change de cible.
- Résultat : **10 réussites sur 10** avec le cube placé au hasard.

```bash
ros2 service call /pick_and_place std_srvs/srv/Trigger   # lance la recette
python3 /ros2_ws/tools/trial_pick_place.py 10            # 10 essais au hasard, notés tout seuls
```

---

## 8. L'organisation de l'équipe
- Montre le dépôt : `git log --graph --all`. On doit voir les commits de chacun.
- Explique qui a fait quoi : le driver et Docker, l'IK et le brain, la calibration et la perception, la phase 2.
- Le message important : **chacun comprend toute la chaîne**, de la caméra jusqu'au moteur.

---

## 9. Phase 2 : la planche Ouija
On pose une **question** au robot et il **répond en épelant** des lettres sur une planche Ouija :
1. Un petit cerveau artificiel, le **LLM** qwen2.5 qui tourne sur l'ordi sans internet, choisit une réponse courte. Exemple : « OUI ».
2. La caméra trouve la planche grâce aux 4 marqueurs ArUco dans les coins, et la planchette, un anneau bleu, grâce à sa couleur.
3. Le bras met le bout de sa pince dans le trou de l'anneau et le **traîne** jusqu'à la lettre.
4. La caméra vérifie : si l'anneau est à plus de 8 mm de la lettre, le bras corrige.
5. Il passe à la lettre suivante, puis va sur FIN.

```bash
ros2 service call /ouija/ask so101_interfaces/srv/Ask "{question: 'est-ce que tu es vivant ?'}"
```

C'est la même chose que la phase 1 (caméra, calibration, IK), réutilisée pour un nouveau jeu.

---

**Pour t'entraîner :** pour chaque point, essaie de l'expliquer en une phrase sans regarder, puis tape la commande correspondante dans le simulateur.