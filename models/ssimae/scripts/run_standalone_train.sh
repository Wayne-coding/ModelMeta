















if [ $
then
    echo "Usage: bash run_standalone_train.sh [CONFIG_PATH] [DEVICE_ID]"
exit 1
fi
get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

CONFIG_PATH=$(get_real_path $1)
export DEVICE_ID=$2
TRAIN_PATH=train_`echo $CONFIG_PATH | rev | cut -d '/' -f 1 | rev | awk -F "_config.yaml" '{print $1}'`
if [ -d $TRAIN_PATH ];
then
    rm -rf $TRAIN_PATH
fi
mkdir $TRAIN_PATH
cd $TRAIN_PATH/ || exit
python ../train.py --config_path=$CONFIG_PATH > train.log 2>&1 &
cd ..
