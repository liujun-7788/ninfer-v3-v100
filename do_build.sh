cd /data/deploy/ninfer-test/build-v100 && setsid ninja apps/ninfer-serve > /data/deploy/ninfer-test/binder_build.log 2>&1 < /dev/null & disown
echo BUILD-FIRED