for i in 0 1 2 3
do
    python predict.py --test_all --count $i --dataset db5_test_unbound
done
