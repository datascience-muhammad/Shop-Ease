# Next Steps: ShopEase 9-Video Series

## Phase 1: Repository Finalization (Ready)

- [x] Create `00_Data_Understanding.ipynb` for Video 2.
- [x] Refactor all notebooks for professional look (no emojis, simple text).
- [x] Update `README.md` and `TEACHING_PATH.md` for 9-video structure.
- [x] Polish `dashboards/streamlit_app.py` UI.

## Phase 2: Content Creation (Next Weeks)

### 1. Scripting (Week 1)

- [ ] Draft 5-minute scripts for each of the 9 videos.
- [ ] Align script key points with code comments in `src/`.
- [ ] Prepare "Golden Code Snippets" for each video's main demonstration.

### 2. Video Recording Schedule

| Video | Title              | Main Artifact                               |
| ----- | ------------------ | ------------------------------------------- |
| 1     | Overview & Setup   | `README.md`                                 |
| 2     | Data Understanding | `00_Data_Understanding.ipynb`               |
| 3     | Data Cleaning      | `01_Data_Cleaning.ipynb`                    |
| 4     | Baseline Modeling  | `src/train.py` (train method)               |
| 5     | BERT Theory        | High-level diagrams                         |
| 6     | BERT Fine-tuning   | `02_Model_Training.ipynb`                   |
| 7     | Evaluation         | `src/evaluate.py`                           |
| 8     | Prediction API     | `03_RealTime_Inference.ipynb`               |
| 9     | Dashboard          | `streamlit run dashboards/streamlit_app.py` |

### 3. Final Validation

- [ ] Run `python run_all.py` one last time on a clean environment.
- [ ] Verify BERT model saves correctly to `models/bert/`.
- [ ] Check Streamlit "Batch Upload" works with `data/demo_reviews.csv`.

---

**Status:** Codebase is 100% ready for recording. Emojis removed, structure simplified, 9-video logic integrated.
