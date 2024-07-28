for ((i=0; i<=4000; i+=100)); do 
    sbatch --export=startAt=$i -p single corruptions_sceneflow.sh ; 
done