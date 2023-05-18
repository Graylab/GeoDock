DIR="db5_test_unbound_0 db5_test_unbound_1 db5_test_unbound_2 db5_test_unbound_3"
#DIR="db5_test_bound_0 db5_test_bound_1 db5_test_bound_2 db5_test_bound_3"
for dir in $DIR
do
    for f in ../predictions/model3/$dir/*.pdb
    do
        name=$(basename $f)
        filename=${name:0:4}_gt.pdb
        /home/lchu11/scr4_jgray21/lchu11/DockQ/DockQ.py $f ../ground_truth/db5_test_bound/$filename -short >> $dir.txt
    done
done
