echo "Run data preprocessing"
cd src
python build_chord_profile.py
python chord_processor.py
python mlu_processor.py
python build_vocab.py
python convert_to_remi.py
python prepare_data.py
echo "Done data preprocessing"