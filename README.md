# BEI_N7_NACA2024

  Qu'est-ce que GitHub ?

GitHub est une plateforme de collaboration qui permet aux développeurs de gérer et de versionner leur code. Il repose sur Git, un système de gestion de version distribué.

Principaux concepts :
Repository (Repo) : Un dossier contenant vos fichiers, y compris l'historique des modifications.
Commit : Un enregistrement de modifications apportées aux fichiers du repo.
Branch : Une version parallèle du code permettant de développer des fonctionnalités sans affecter la version principale (généralement main).
Pull Request (PR) : Une demande pour intégrer les modifications d'une branche dans une autre.
Fork : Une copie d'un repo que vous pouvez modifier sans affecter le repo original.
Clone : Télécharger un repo localement.


  Choisir un terminal pour exécuter les commandes Git
Pour utiliser Git et exécuter des commandes comme git push, git pull ou git status, vous avez besoin d'un terminal (ou ligne de commande). Selon votre système d'exploitation, les options peuvent varier :

Windows :
Utilisez Git Bash (installé avec Git pour Windows) qui simule un terminal Linux et fonctionne très bien avec Git.
Windows Terminal ou Command Prompt (cmd) fonctionnent aussi, mais Git Bash est plus puissant et plus flexible.
macOS : Utilisez l’application Terminal qui est installée par défaut sur tous les Macs. Alternativement, vous pouvez utiliser des émulateurs comme iTerm.

Linux : Utilisez simplement le Terminal (ou shell) déjà installé dans la majorité des distributions Linux.
Une fois que vous avez ouvert le terminal ou Git Bash, vous pouvez vérifier si Git est bien installé avec cette commande :

git --version
Cela affichera la version de Git installée si tout fonctionne correctement.

  Créer et gérer une nouvelle branche : de A à Z

Qu'est-ce qu'une branche ?

Dans Git, une branche est une version parallèle de votre code. Cela vous permet de travailler sur une fonctionnalité spécifique ou une correction de bug sans affecter le code principal (généralement dans la branche main ou master).

Voici un guide détaillé pour créer et gérer une nouvelle branche à partir d'un dépôt existant.

  Cloner un dépôt Git (si ce n'est pas déjà fait)
  
Si vous travaillez sur un projet existant sur GitHub et que vous n'avez pas encore de copie locale, vous devez cloner le dépôt. Cela vous permet de télécharger une copie locale du code depuis GitHub.
git clone https://github.com/username/repository.git

Remplacez username par le nom d'utilisateur du propriétaire du dépôt, et repository par le nom du dépôt.
Cela crée un dossier local avec tout le contenu du dépôt distant.

  Se déplacer dans le répertoire cloné
  
Entrez dans le répertoire cloné avec la commande cd :
cd repository

(Remplacez repository par le nom du projet cloné.)

  Créer une nouvelle branche
Pour créer une nouvelle branche (par exemple, pour une nouvelle fonctionnalité), suivez ces étapes :

Vérifiez sur quelle branche vous êtes :

git branch
Cela vous montrera les branches existantes et mettra en évidence celle sur laquelle vous êtes actuellement.

Créez une nouvelle branche : Utilisez la commande suivante pour créer une nouvelle branche (par exemple, pour une fonctionnalité spécifique) :

git checkout -b nom-de-la-branche
nom-de-la-branche : Donnez un nom descriptif à votre branche, comme feature-nouvelle-fonctionnalite ou bugfix-correction-issue.
Par exemple :


  Ajouter et committer des modifications

Après avoir modifié ou ajouté des fichiers, exécutez :

git add .

Ensuite, validez les modifications :

git commit -m "Message décrivant les changements"

A tout moment, il est possible de check le statut des modifications avec la commande : 

git status

  Ajouter les modifications sur la branche principale

Une fois que vous avez validé vos modifications localement avec git commit, vous pouvez les "pousser" (push) vers le dépôt distant sur GitHub pour les partager avec d'autres collaborateurs. Il ne s'agit pas de rajouter directement les modifications à la branche principale mais seulement d'ajouter des modifications locales à une branche du dépot. (Attention de ne pas push sur Main)

Commande pour pousser :

git push origin branch-name

origin : Il s'agit du dépôt distant (GitHub dans ce cas). Par défaut, le dépôt distant s'appelle "origin".
branch-name : Remplacez ceci par le nom de votre branche. Il est important de pousser les modifications vers une branche spécifique et non directement sur main (plus de détails ci-dessous).


   PULL REQUEST 

Demander une fusion via une Pull Request (PR)
Une fois que vous avez poussé vos modifications vers une branche sur GitHub, vous pouvez soumettre une Pull Request (PR) pour demander la fusion de vos modifications dans une autre branche, souvent la branche principale (main).

Qu'est-ce qu'une Pull Request ?

Une pull request est une demande pour que quelqu'un d'autre (souvent un mainteneur du projet) examine vos changements avant de les intégrer à la branche cible (souvent main). Cela permet aux équipes de réviser le code, d'avoir des discussions, et de s'assurer que les modifications sont conformes aux normes du projet avant qu'elles ne soient fusionnées dans le code principal.

Étapes pour créer une Pull Request :

Pousser votre branche vers GitHub (comme vu dans la section précédente).
Accédez à votre dépôt sur GitHub et cliquez sur "Compare & pull request" qui apparaîtra après que vous ayez poussé une nouvelle branche.
Sur la page de la PR :
Titre : Décrivez brièvement la modification.
Description : Expliquez en détail ce que vous avez fait, pourquoi ces modifications sont nécessaires, et toute autre information utile.
Comparaison de branches : Assurez-vous que vous comparez bien la bonne branche (ex. feature-nouvelle-fonctionnalite) avec la branche dans laquelle vous souhaitez intégrer les modifications (ex. main).
Cliquez sur "Create Pull Request".
Une fois la PR soumise, d'autres membres de l'équipe peuvent la réviser, poser des questions, demander des modifications, ou l'approuver. Une fois approuvée, elle peut être fusionnée (merge) dans la branche cible.


Vour pourrez trouver énormément de documentation sur internet, voici quelques sites utiles : 

https://git-scm.com/book/en/v2

https://docs.github.com/fr















