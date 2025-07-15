for batch_number in {0..3}; do
    sbatch scripts/r1/pass_at_k.sh $batch_number
done
