for batch_number in {1..3}; do
    sbatch scripts/r1/pass_at_k_direct.sh $batch_number
done
