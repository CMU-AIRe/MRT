for batch_number in {0..3}; do
    sbatch scripts/r1/pass_at_k_given_prefix.sh $batch_number
done
