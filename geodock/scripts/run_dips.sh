DIR="dips_test_0 dips_test_1 dips_test_2 dips_test_3"
for dir in $DIR
do
    for f in ../predictions/$dir/*.pdb
    do
        name=$(basename $f)
        filename=${name:0:-6}.pdb
        /home/lchu11/scr4_jgray21/lchu11/DockQ/DockQ.py $f ../ground_truth/dips_test/$filename -short >> $dir.txt
    done
done
