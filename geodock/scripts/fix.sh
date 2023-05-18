for f in ../pdbs_400/*.pdb
do 
    name=$(basename $f)
    filename=${name:0:4}_b.pdb
    /home/lchu11/scr4_jgray21/lchu11/DockQ/scripts/fix_numbering.pl $f /home/lchu11/scr4_jgray21/lchu11/Docking-dev/data/db5_native/$filename
done
