import os
import time
from pathlib import Path
from typing import List, Tuple

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import pandas as pd
import streamlit as st

from contextual_summarizer import summarize_file


MODEL_PATH = Path("models/contextual_ranker.pt")
UPLOAD_DIR = Path("data/uploads")


st.set_page_config(
    page_title="Offline Contextual-Ranking Summarizer",
    layout="wide",
)


@st.cache_resource
def load_model_status() -> bool:
    """
    Confirms that the trained learned contextual-ranker checkpoint
    exists before a user uploads a file.
    """
    return MODEL_PATH.exists()


def summarize_uploaded_file(
    input_path: Path,
    top_k: int,
) -> Tuple[List[str], pd.DataFrame]:
    """
    Runs the final proposed summarizer.

    summarize_file loads the trained contextual-ranking hybrid from
    mamba_summarizer.py and returns summary sentences plus score data.
    """
    return summarize_file(
        input_path=str(input_path),
        top_k=top_k,
    )


st.title("Lightweight Offline Extractive Summarizer")

st.caption(
    "Final proposed model: learned contextual sentence ranking + "
    "MiniLM semantic relevance + TF-IDF keyword importance + "
    "sentence-position relevance"
)

st.info(
    "This application runs locally on CPU. It uses the trained "
    "contextual-ranking checkpoint and does not silently fall back "
    "to a semantic baseline."
)

if not load_model_status():
    st.error(
        "The trained model checkpoint was not found: "
        f"`{MODEL_PATH}`"
    )

    st.warning(
        "Run the following command from the project folder before "
        "starting the app:"
    )

    st.code(
        "py train_mamba.py",
        language="powershell",
    )

    st.stop()

st.success(
    f"Trained checkpoint found: {MODEL_PATH}"
)

uploaded = st.file_uploader(
    "Choose a TXT or PDF document",
    type=["txt", "pdf"],
)

top_k = st.slider(
    "Number of summary sentences",
    min_value=1,
    max_value=10,
    value=3,
)

if uploaded is not None:
    UPLOAD_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    safe_filename = Path(uploaded.name).name

    input_path = UPLOAD_DIR / safe_filename

    input_path.write_bytes(
        uploaded.getvalue()
    )

    st.write(f"**Uploaded file:** {safe_filename}")

    st.write(f"**Summary length:** {top_k} sentence(s)")

    try:
        start_time = time.perf_counter()

        with st.spinner(
            "Loading the contextual-ranking model and generating "
            "the summary..."
        ):
            summary, score_details = summarize_uploaded_file(
                input_path=input_path,
                top_k=top_k,
            )

        elapsed_time = time.perf_counter() - start_time

        if not summary:
            st.warning(
                "No usable sentences were found in the uploaded "
                "document."
            )
        else:
            st.subheader("Generated Extractive Summary")

            for number, sentence in enumerate(
                summary,
                start=1,
            ):
                st.write(f"**{number}.** {sentence}")

            summary_text = "\n".join(summary)

            st.download_button(
                label="Download Summary as TXT",
                data=summary_text,
                file_name="contextual_ranking_summary.txt",
                mime="text/plain",
            )

            st.caption(
                f"Processing time: {elapsed_time:.2f} seconds"
            )

            if not score_details.empty:
                with st.expander(
                    "View sentence-ranking details"
                ):
                    st.dataframe(
                        score_details,
                        use_container_width=True,
                        hide_index=True,
                    )

                score_csv = score_details.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    label="Download Sentence Scores as CSV",
                    data=score_csv,
                    file_name="sentence_ranking_scores.csv",
                    mime="text/csv",
                )

    except FileNotFoundError as error:
        st.error(
            "The trained ranking checkpoint could not be loaded."
        )

        st.code(
            "py train_mamba.py",
            language="powershell",
        )

        st.exception(error)

    except Exception as error:
        st.error(
            f"Could not summarize the uploaded document: {error}"
        )

        st.exception(error)