cd new_optimized
for entry in ./*
do
    number=$(basename $entry)
    new=$(($number + 394))
    mv $entry "$new.mol"
done
cd ..
