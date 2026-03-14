#!/bin/bash
# Download BSL sign videos from SignBSL.com for batch testing
# Videos are stored in frontend/test-videos/ and served via our CloudFront

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VIDEO_DIR="$SCRIPT_DIR/../frontend/test-videos"
mkdir -p "$VIDEO_DIR"

# Map: filename|source_url
# Only include signs that exist in our sign-dictionary.js (270 signs)
# Exclude dubious mappings where the video filename suggests a different sign
VIDEOS=(
    # GREETINGS & SOCIAL
    "HELLO|https://media.signbsl.com/videos/bsl/signstation/HI.mp4"
    "GOODBYE|https://media.signbsl.com/videos/bsl/signstation/goodbye.mp4"
    "WELCOME|https://media.signbsl.com/videos/bsl/signstation/welcome.mp4"
    "YES|https://media.signbsl.com/videos/bsl/signstation/yes.mp4"
    "NO|https://media.signbsl.com/videos/bsl/signstation/no.mp4"
    "PLEASE|https://media.signbsl.com/videos/bsl/signstation/please.mp4"
    "THANK-YOU|https://media.signbsl.com/videos/bsl/signstation/thank%20you.mp4"
    "SORRY|https://media.signbsl.com/videos/bsl/signstation/sorry.mp4"
    "CORRECT|https://media.signbsl.com/videos/bsl/signstation/right.mp4"
    "WRONG|https://media.signbsl.com/videos/bsl/signstation/wrong.mp4"

    # QUESTION WORDS
    "WHAT|https://media.signbsl.com/videos/bsl/signstation/what.mp4"
    "WHERE|https://media.signbsl.com/videos/bsl/signstation/where.mp4"
    "WHEN|https://media.signbsl.com/videos/bsl/signstation/when.mp4"
    "WHO|https://media.signbsl.com/videos/bsl/signstation/WHO.mp4"
    "WHY|https://media.signbsl.com/videos/bsl/signstation/why.mp4"
    "HOW|https://media.signbsl.com/videos/bsl/signstation/how.mp4"

    # PEOPLE
    "MAN|https://media.signbsl.com/videos/bsl/signstation/man.mp4"
    "WOMAN|https://media.signbsl.com/videos/bsl/signstation/woman.mp4"
    "BOY|https://media.signbsl.com/videos/bsl/signstation/boy.mp4"
    "GIRL|https://media.signbsl.com/videos/bsl/signstation/girl.mp4"
    "CHILD|https://media.signbsl.com/videos/bsl/signstation/child.mp4"
    "MOTHER|https://media.signbsl.com/videos/bsl/signstation/mother.mp4"
    "FATHER|https://media.signbsl.com/videos/bsl/signstation/father.mp4"
    "FAMILY|https://media.signbsl.com/videos/bsl/signstation/family.mp4"
    "FRIEND|https://media.signbsl.com/videos/bsl/signstation/friend.mp4"
    "DOCTOR|https://media.signbsl.com/videos/bsl/signstation/doctor.mp4"
    "TEACHER|https://media.signbsl.com/videos/bsl/signstation/teacher.mp4"
    "POLICE|https://media.signbsl.com/videos/bsl/signstation/police.mp4"
    "DEAF|https://media.signbsl.com/videos/bsl/signstation/deaf.mp4"
    "HEARING|https://media.signbsl.com/videos/bsl/signstation/101-06-1372.mp4"

    # COMMON VERBS
    "GO|https://media.signbsl.com/videos/bsl/signstation/go.mp4"
    "COME|https://media.signbsl.com/videos/bsl/signstation/come.mp4"
    "STOP|https://media.signbsl.com/videos/bsl/signstation/end.mp4"
    "WAIT|https://media.signbsl.com/videos/bsl/signstation/wait.mp4"
    "SIT|https://media.signbsl.com/videos/bsl/signstation/sit.mp4"
    "STAND|https://media.signbsl.com/videos/bsl/signstation/101-06-1113.mp4"
    "WALK|https://media.signbsl.com/videos/bsl/signstation/walk.mp4"
    "RUN|https://media.signbsl.com/videos/bsl/signstation/run.mp4"
    "EAT|https://media.signbsl.com/videos/bsl/signstation/eat.mp4"
    "DRINK|https://media.signbsl.com/videos/bsl/signstation/drink.mp4"
    "SLEEP|https://media.signbsl.com/videos/bsl/signstation/sleep.mp4"
    "WORK|https://media.signbsl.com/videos/bsl/signstation/work.mp4"
    "HELP|https://media.signbsl.com/videos/bsl/signstation/help.mp4"
    "WANT|https://media.signbsl.com/videos/bsl/signstation/want.mp4"
    "LIKE|https://media.signbsl.com/videos/bsl/signstation/like.mp4"
    "LOVE|https://media.signbsl.com/videos/bsl/signstation/love.mp4"
    "KNOW|https://media.signbsl.com/videos/bsl/signstation/know.mp4"
    "UNDERSTAND|https://media.signbsl.com/videos/bsl/signstation/understand.mp4"
    "LEARN|https://media.signbsl.com/videos/bsl/signstation/learn.mp4"
    "OPEN|https://media.signbsl.com/videos/bsl/signstation/open.mp4"
    "CLOSE|https://media.signbsl.com/videos/bsl/signstation/close.mp4"
    "START|https://media.signbsl.com/videos/bsl/signstation/get.mp4"
    "FINISH|https://media.signbsl.com/videos/bsl/signstation/end.mp4"
    "CLEAN|https://media.signbsl.com/videos/bsl/signstation/clean.mp4"
    "SIGN|https://media.signbsl.com/videos/bsl/signstation/sign.mp4"
    "BOOK|https://media.signbsl.com/videos/bsl/signstation/book.mp4"
    "AGAIN|https://media.signbsl.com/videos/bsl/signstation/again.mp4"
    "MORE|https://media.signbsl.com/videos/bsl/signstation/more.mp4"

    # ADJECTIVES & DESCRIPTIONS
    "GOOD|https://media.signbsl.com/videos/bsl/signstation/good.mp4"
    "BAD|https://media.signbsl.com/videos/bsl/signstation/bad.mp4"
    "BIG|https://media.signbsl.com/videos/bsl/signstation/big.mp4"
    "SMALL|https://media.signbsl.com/videos/bsl/signstation/small.mp4"
    "HAPPY|https://media.signbsl.com/videos/bsl/signstation/happy.mp4"
    "SAD|https://media.signbsl.com/videos/bsl/signstation/101-06-605.mp4"
    "HUNGRY|https://media.signbsl.com/videos/bsl/signstation/hungry.mp4"
    "THIRSTY|https://media.signbsl.com/videos/bsl/signstation/thirsty.mp4"
    "TIRED|https://media.signbsl.com/videos/bsl/signstation/tired.mp4"
    "HOT|https://media.signbsl.com/videos/bsl/signstation/hot.mp4"
    "COLD|https://media.signbsl.com/videos/bsl/signstation/101-06-1008.mp4"
    "BEAUTIFUL|https://media.signbsl.com/videos/bsl/signstation/beautiful.mp4"
    "UGLY|https://media.signbsl.com/videos/bsl/signstation/ugly.mp4"
    "NEW|https://media.signbsl.com/videos/bsl/signstation/new.mp4"
    "OLD|https://media.signbsl.com/videos/bsl/signstation/old.mp4"
    "SAME|https://media.signbsl.com/videos/bsl/signstation/same.mp4"
    "DIFFERENT|https://media.signbsl.com/videos/bsl/signstation/different.mp4"
    "IMPORTANT|https://media.signbsl.com/videos/bsl/signstation/important.mp4"
    "FREE|https://media.signbsl.com/videos/bsl/signstation/free.mp4"
    "SICK|https://media.signbsl.com/videos/bsl/signstation/sick.mp4"
    "NICE-TO-MEET-YOU|https://media.signbsl.com/videos/bsl/signstation/nice.mp4"
    "PAIN|https://media.signbsl.com/videos/bsl/gpnhs/mp4/pain.mp4"

    # TIME
    "TODAY|https://media.signbsl.com/videos/bsl/signstation/today.mp4"
    "TOMORROW|https://media.signbsl.com/videos/bsl/signstation/tomorrow.mp4"
    "YESTERDAY|https://media.signbsl.com/videos/bsl/signstation/yesterday.mp4"
    "MORNING|https://media.signbsl.com/videos/bsl/signstation/good%20morning.mp4"
    "AFTERNOON|https://media.signbsl.com/videos/bsl/signstation/good-afternoon.mp4"
    "EVENING|https://media.signbsl.com/videos/bsl/signstation/evening.mp4"
    "NIGHT|https://media.signbsl.com/videos/bsl/signstation/dark.mp4"
    "WEEK|https://media.signbsl.com/videos/bsl/signstation/week.mp4"
    "MONTH|https://media.signbsl.com/videos/bsl/signstation/month.mp4"
    "YEAR|https://media.signbsl.com/videos/bsl/signstation/year.mp4"

    # PLACES & TRAVEL
    "HOUSE|https://media.signbsl.com/videos/bsl/signstation/house.mp4"
    "SCHOOL|https://media.signbsl.com/videos/bsl/signstation/school.mp4"
    "HOSPITAL|https://media.signbsl.com/videos/bsl/deafway/mp4/hospital.mp4"
    "CAR|https://media.signbsl.com/videos/bsl/signstation/car.mp4"
    "BUS|https://media.signbsl.com/videos/bsl/signstation/bus.mp4"
    "TRAIN|https://media.signbsl.com/videos/bsl/signstation/101-06-620.mp4"

    # FOOD & DRINK
    "FOOD|https://media.signbsl.com/videos/bsl/signstation/food.mp4"
    "WATER|https://media.signbsl.com/videos/bsl/signstation/water.mp4"
    "TEA|https://media.signbsl.com/videos/bsl/signstation/tea.mp4"
    "COFFEE|https://media.signbsl.com/videos/bsl/signstation/coffee.mp4"
    "MILK|https://media.signbsl.com/videos/bsl/signstation/milk.mp4"

    # COLOURS
    "RED|https://media.signbsl.com/videos/bsl/signstation/red.mp4"
    "BLUE|https://media.signbsl.com/videos/bsl/signstation/blue.mp4"
    "GREEN|https://media.signbsl.com/videos/bsl/signstation/green.mp4"
    "YELLOW|https://media.signbsl.com/videos/bsl/signstation/yellow.mp4"
    "WHITE|https://media.signbsl.com/videos/bsl/signstation/white.mp4"
    "BLACK|https://media.signbsl.com/videos/bsl/signstation/black.mp4"

    # ANIMALS
    "DOG|https://media.signbsl.com/videos/bsl/signstation/dog.mp4"
    "CAT|https://media.signbsl.com/videos/bsl/signstation/cat.mp4"

    # MISCELLANEOUS
    "MONEY|https://media.signbsl.com/videos/bsl/signstation/money.mp4"
    "COMPUTER|https://media.signbsl.com/videos/bsl/signstation/computer.mp4"
    "PHONE|https://media.signbsl.com/videos/bsl/signstation/telephone.mp4"
    "NAME|https://media.signbsl.com/videos/bsl/signstation/name.mp4"
    "LANGUAGE|https://media.signbsl.com/videos/bsl/signstation/language.mp4"
    "QUESTION|https://media.signbsl.com/videos/bsl/signstation/question.mp4"
    "ANSWER|https://media.signbsl.com/videos/bsl/signstation/answer.mp4"
    "PROBLEM|https://media.signbsl.com/videos/bsl/signstation/problem.mp4"
    "MEETING|https://media.signbsl.com/videos/bsl/signstation/meeting.mp4"
    "NUMBER|https://media.signbsl.com/videos/bsl/signstation/figure.mp4"

    # HIGHER QUALITY from BSL SignBank (UCL) — supplement
    "HELLO-HQ|https://bslsignbank.ucl.ac.uk/media/bsl-video/HE/HELLO.mp4"
    "YES-HQ|https://bslsignbank.ucl.ac.uk/media/bsl-video/YE/YES.mp4"
    "NO-HQ|https://bslsignbank.ucl.ac.uk/media/bsl-video/NO/NO.mp4"
    "PLEASE-HQ|https://bslsignbank.ucl.ac.uk/media/bsl-video/PL/PLEASE.mp4"
    "SORRY-HQ|https://bslsignbank.ucl.ac.uk/media/bsl-video/SO/SORRY.mp4"
    "HELP-HQ|https://bslsignbank.ucl.ac.uk/media/bsl-video/HE/HELP.mp4"
    "GOOD-HQ|https://bslsignbank.ucl.ac.uk/media/bsl-video/FA/FANTASTIC.mp4"
    "BAD-HQ|https://bslsignbank.ucl.ac.uk/media/bsl-video/BA/BAD.mp4"
    "HAPPY-HQ|https://bslsignbank.ucl.ac.uk/media/bsl-video/HA/HAPPY.mp4"
    "FAMILY-HQ|https://bslsignbank.ucl.ac.uk/media/bsl-video/FA/FAMILY.mp4"
    "MOTHER-HQ|https://bslsignbank.ucl.ac.uk/media/bsl-video/MO/MOTHER.mp4"
    "FRIEND-HQ|https://bslsignbank.ucl.ac.uk/media/bsl-video/FR/FRIEND.mp4"
    "SCHOOL-HQ|https://bslsignbank.ucl.ac.uk/media/bsl-video/SC/SCHOOL.mp4"
    "WORK-HQ|https://bslsignbank.ucl.ac.uk/media/bsl-video/WO/WORK.mp4"
    "HOUSE-HQ|https://bslsignbank.ucl.ac.uk/media/bsl-video/HO/HOUSE.mp4"
    "MONEY-HQ|https://bslsignbank.ucl.ac.uk/media/bsl-video/MO/MONEY.mp4"
    "KNOW-HQ|https://bslsignbank.ucl.ac.uk/media/bsl-video/KN/KNOW.mp4"
    "LEARN-HQ|https://bslsignbank.ucl.ac.uk/media/bsl-video/LE/LEARN.mp4"
    "DRINK-HQ|https://bslsignbank.ucl.ac.uk/media/bsl-video/DR/DRINK.mp4"
    "SLEEP-HQ|https://bslsignbank.ucl.ac.uk/media/bsl-video/SL/SLEEP.mp4"
    "BIG-HQ|https://bslsignbank.ucl.ac.uk/media/bsl-video/BI/BIG.mp4"
    "RED-HQ|https://bslsignbank.ucl.ac.uk/media/bsl-video/RE/RED.mp4"
    "BLUE-HQ|https://bslsignbank.ucl.ac.uk/media/bsl-video/BL/BLUE.mp4"
    "GREEN-HQ|https://bslsignbank.ucl.ac.uk/media/bsl-video/GR/GREEN.mp4"
    "WHITE-HQ|https://bslsignbank.ucl.ac.uk/media/bsl-video/WH/WHITE.mp4"
    "BLACK-HQ|https://bslsignbank.ucl.ac.uk/media/bsl-video/BL/BLACK.mp4"
    "DOG-HQ|https://bslsignbank.ucl.ac.uk/media/bsl-video/DO/DOG.mp4"
    "CAT-HQ|https://bslsignbank.ucl.ac.uk/media/bsl-video/CA/CAT.mp4"
    "DEAF-HQ|https://bslsignbank.ucl.ac.uk/media/bsl-video/DE/DEAF.mp4"
    "WANT-HQ|https://bslsignbank.ucl.ac.uk/media/bsl-video/WA/WANT.mp4"
    "WEEK-HQ|https://bslsignbank.ucl.ac.uk/media/bsl-video/WE/WEEK.mp4"
    "MORNING-HQ|https://bslsignbank.ucl.ac.uk/media/bsl-video/MO/MORNING.mp4"
    "DAY-HQ|https://bslsignbank.ucl.ac.uk/media/bsl-video/DA/DAY.mp4"
)

echo "Downloading ${#VIDEOS[@]} test videos..."
FAILED=0
SUCCESS=0

for entry in "${VIDEOS[@]}"; do
    IFS='|' read -r name url <<< "$entry"
    outfile="$VIDEO_DIR/${name}.mp4"

    if [ -f "$outfile" ]; then
        echo "  [skip] $name (already exists)"
        ((SUCCESS++))
        continue
    fi

    echo -n "  [download] $name ... "
    if curl -sS -L -o "$outfile" "$url" 2>/dev/null; then
        # Verify it's actually a video (check file size > 1KB)
        size=$(stat -f%z "$outfile" 2>/dev/null || stat -c%s "$outfile" 2>/dev/null)
        if [ "$size" -gt 1024 ]; then
            echo "OK (${size} bytes)"
            ((SUCCESS++))
        else
            echo "FAIL (too small: ${size} bytes)"
            rm -f "$outfile"
            ((FAILED++))
        fi
    else
        echo "FAIL (download error)"
        rm -f "$outfile"
        ((FAILED++))
    fi
done

echo ""
echo "Done: $SUCCESS downloaded, $FAILED failed"
echo "Videos in: $VIDEO_DIR"
ls -la "$VIDEO_DIR" | head -5
echo "... ($(ls "$VIDEO_DIR" | wc -l) files total)"
