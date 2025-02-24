

conda create -n ./crewai-env python=3.12

현재 디렉토리에 환경을 만들고 싶다면
conda create -p ./crewai-env python=3.12

pip install 'crewai[tools]'


conda activate ./crewai-env