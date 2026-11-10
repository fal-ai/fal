# Run as: sh -c "<this file>" sh FILE UNTIL KEPT_EXIT
# Writes UNTIL (epoch seconds) to FILE, unless FILE already holds a later time:
# then it changes nothing and exits with status KEPT_EXIT.
old=$(cat "$1" 2>/dev/null || echo 0)
if [ "${old%.*}" -gt "$2" ] 2>/dev/null; then
    exit "$3"
fi
echo "$2" > "$1"
