if not exist ..\build-arm mkdir ..\build-arm
cd ..\build-arm
if errorlevel 1 exit
cd ../build-arm || exit

cmake .. ^
-DCMAKE_TOOLCHAIN_FILE=%NDK_ROOT%\build\cmake\android.toolchain.cmake ^
-DCMAKE_BUILD_TYPE=Release ^
-DANDROID_ABI="arm64-v8a" ^
-DANDROID_NATIVE_API_LEVEL=android-28 ^
-DCMAKE_CXX_FLAGS="-march=armv8.2-a+dotprod" ^
-DNATIVE_LIBRARY_OUTPUT=. -DNATIVE_INCLUDE_OUTPUT=. %1 %2 %3 ^
-DDEBUG=OFF ^
-DTEST=OFF ^
-DARM=ON ^
-DAPK=OFF ^
-GNinja

Ninja -j4