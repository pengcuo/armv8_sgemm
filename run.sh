../script/build_64.sh

adb push test_integral_meanfilter /data/local/tmp/integral

adb shell LD_LIBRARY_PATH=/data/local/tmp/integral /data/local/tmp/integral/test_integral_meanfilter
