while getopts 'dpf:v' flag; do
  case "${flag}" in
    d) d_flag='true' ;;
    p) d_flag='false' ;;
    f) files="${OPTARG}" ;;
  esac
done
echo $files
echo $d_flag

if $d_flag = "true" 
then
  python main.py $files -d
else
  python main.py $files -p
fi


