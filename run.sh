../script/build_64.sh

adb push test_armv8_quadrbits_gemm /data/local/tmp/int4

adb shell LD_LIBRARY_PATH=/data/local/tmp/int4 /data/local/tmp/int4/test_armv8_quadrbits_gemm
