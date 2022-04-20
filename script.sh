while getopts 'df:v' flag; do
  case "${flag}" in
    d) d_flag='true' ;;
    f) files="${OPTARG}" ;;
    *) print_usage
       exit 1 ;;
  esac
done
echo $files
echo $d_flag
if d_flag="true" 
then
  python main.py $files -d
else
  python main.py $files -p
fi

