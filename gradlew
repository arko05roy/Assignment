#!/bin/sh

APP_HOME=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
CLASSPATH=$APP_HOME/gradle/wrapper/gradle-wrapper.jar
ANDROID_STUDIO_JAVA='/Applications/Android Studio.app/Contents/jbr/Contents/Home/bin/java'

if [ ! -f "$CLASSPATH" ]; then
  echo "Missing Gradle wrapper JAR: $CLASSPATH" >&2
  exit 1
fi

if [ -x "$ANDROID_STUDIO_JAVA" ]; then
  exec "$ANDROID_STUDIO_JAVA" -classpath "$CLASSPATH" org.gradle.wrapper.GradleWrapperMain "$@"
fi

exec java -classpath "$CLASSPATH" org.gradle.wrapper.GradleWrapperMain "$@"
