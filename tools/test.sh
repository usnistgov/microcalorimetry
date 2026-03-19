if [[ $1 == "open" ]]
then
    set -e
    start htmlcov/index.html

else
    set -e
	echo "Running pytest and passing any additional arguments..."
    echo " useful extra args :"
    echo " open : passed as first argument opens a test report."
    echo " -x : stop after first failure"
    uv run pytest tests --cov=microcalorimetry tests/ src/  --junitxml=tests/report.xml --cov-report html --cov-report term  --doctest-modules "$@"
fi
exit 0