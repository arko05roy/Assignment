#!/bin/sh
set -eu

ANDROID_SDK_ROOT='/Users/arkoroy/Library/Android/sdk'
ADB="$ANDROID_SDK_ROOT/platform-tools/adb"
EMULATOR="$ANDROID_SDK_ROOT/emulator/emulator"
APK='app/build/outputs/apk/debug/app-debug.apk'

if ! "$EMULATOR" -list-avds | grep -qx 'FaceReel_Demo'; then
  echo 'Missing FaceReel_Demo AVD. Open Android Studio > Device Manager and create it first.' >&2
  exit 1
fi

if ! "$ADB" devices | grep -qE '^emulator-[0-9]+[[:space:]]+device$'; then
  "$EMULATOR" -avd FaceReel_Demo -camera-back webcam1 -no-snapshot-load -gpu swiftshader_indirect &
fi

"$ADB" wait-for-device
until [ "$("$ADB" shell getprop sys.boot_completed | tr -d '\r')" = '1' ]; do sleep 2; done
./gradlew assembleDebug --no-daemon
"$ADB" install -r "$APK"
"$ADB" shell pm grant com.example.facereel android.permission.CAMERA
"$ADB" shell am start -n com.example.facereel/.MainActivity
