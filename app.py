# ---------------------------------------------------------#
# Paytm Logo Visibility Analysis in ICC Cricket Tournament #
# ---------------------------------------------------------#

import streamlit as st
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO
from fpdf import FPDF
import os # Import os for file cleanup
import logging # For better error reporting

# Configure logging for the application
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------

st.set_page_config(
    page_title="Paytm Logo Visibility Analyzer",
    page_icon="📺",
    layout="wide"
)

# ------------------------------------------------
# CONSTANTS & COLOR PALETTE
# ------------------------------------------------
# Define a professional and consistent color palette
COLOR_FULLY_VISIBLE = '#4CAF50'  # Green
COLOR_PARTIALLY_VISIBLE = '#2196F3' # Blue
COLOR_NOT_VISIBLE = '#9E9E9E'    # Gray
COLOR_ACCENT_PRIMARY = '#1976D2' # Deep Blue

# Model path
MODEL_PATH = "./yolov8m_paytm_logo_optuna2/weights/best.pt"

# YOLO Class IDs (assuming these are fixed from training)
CLASS_ID_FULLY_VISIBLE = 0
CLASS_ID_PARTIALLY_VISIBLE = 1

# PDF Report Constants
PAGE_WIDTH_MM = 210 # A4 width in mm
PAGE_HEIGHT_MM = 297 # A4 height in mm
IMAGE_WIDTH_PIE_MM = 120 # Width for pie charts in mm
IMAGE_HEIGHT_PIE_MM = 120 # Height for pie charts in mm (assuming square)
IMAGE_WIDTH_BAR_MM = 180 # Width for bar charts in mm
IMAGE_HEIGHT_BAR_MM = 90 # Height for bar charts in mm (approx, adjust as needed)
MARGIN_BELOW_TITLE_MM = 5 # Small margin in mm between title and image
PDF_TABLE_COL_WIDTH_MM = 60 # Width for columns in the PDF table

# Temporary file names
TEMP_VIDEO_FILE = "temp_video.mp4"
TEMP_PDF_REPORT = "report.pdf"
TEMP_PLOT_FILES = ["pie1.png", "pie2.png", "second_plot.png", "minute_plot.png"]


# ------------------------------------------------
# LOAD MODEL
# ------------------------------------------------

# Check if model file exists
if not os.path.exists(MODEL_PATH):
    st.error(f"Error: Model file not found at '{MODEL_PATH}'. Please ensure the model is in the correct directory.")
    logging.error(f"Model file not found: {MODEL_PATH}")
    st.stop() # Stop the app if the model isn't found

@st.cache_resource # Cache the model loading to avoid reloading on every rerun
def load_yolo_model(path: str) -> YOLO:
    """
    Loads the YOLO model from the specified path and caches it.

    Args:
        path (str): The file path to the YOLO model weights.

    Returns:
        YOLO: The loaded YOLO model object.

    Raises:
        Exception: If the model fails to load.
    """
    try:
        logging.info(f"Attempting to load YOLO model from: {path}")
        model = YOLO(path)
        logging.info("YOLO model loaded successfully.")
        return model
    except Exception as e:
        st.error(f"Failed to load YOLO model: {e}")
        logging.exception(f"Failed to load YOLO model from {path}")
        st.stop()

model = load_yolo_model(MODEL_PATH)

# ------------------------------------------------
# TIME FORMAT UTILITY
# ------------------------------------------------

def seconds_to_hms(seconds: float) -> str:
    """
    Converts total seconds into HH:MM:SS format.

    Args:
        seconds (float): The total number of seconds.

    Returns:
        str: The time formatted as HH:MM:SS.
    """
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

# ------------------------------------------------
# VIDEO ANALYSIS LOGIC
# ------------------------------------------------

def analyze_video(video_path: str) -> pd.DataFrame:
    """
    Analyzes a video for Paytm logo visibility using a YOLO model.
    Processes one frame per second and returns a DataFrame with timestamped visibility counts.

    Args:
        video_path (str): The file path to the video to be analyzed.

    Returns:
        pd.DataFrame: A DataFrame containing 'timestamp', 'fully_visible', and 'partially_visible'
                      columns for each second of the video. Returns an empty DataFrame on error.
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        st.error(f"Error: Could not open video file at {video_path}. Please check the file path and format.")
        logging.error(f"Failed to open video file: {video_path}")
        return pd.DataFrame()

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_video_seconds = total_frames / fps if fps > 0 else 0

    # Analyze one frame per second
    frame_interval = fps if fps > 0 else 1 # Avoid division by zero if fps is somehow 0
    frame_no = 0

    results_list = []

    progress_text = "Analyzing video frames..."
    progress_bar = st.progress(0, text=progress_text)

    logging.info(f"Starting video analysis for {video_path} (FPS: {fps}, Total Frames: {total_frames})")

    while True:
        ret, frame = cap.read()

        if not ret:
            logging.info("End of video stream or read error.")
            break

        # Process only frames at the specified interval (e.g., once per second)
        if frame_no % frame_interval == 0:
            timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            seconds = timestamp_ms / 1000
            timestamp_hms = seconds_to_hms(seconds)

            fully_visible_count = 0
            partially_visible_count = 0

            try:
                # Perform inference
                # imgsz=512 is a common practice for YOLO models, adjust if your model expects different
                # verbose=False to suppress console output from YOLO
                results = model(frame, imgsz=512, verbose=False)

                for r in results:
                    for box in r.boxes:
                        cls = int(box.cls)
                        if cls == CLASS_ID_FULLY_VISIBLE:
                            fully_visible_count += 1
                        elif cls == CLASS_ID_PARTIALLY_VISIBLE:
                            partially_visible_count += 1
            except Exception as e:
                logging.error(f"Error during YOLO inference at frame {frame_no}, timestamp {timestamp_hms}: {e}")
                # Continue processing other frames even if one fails
                pass

            results_list.append(
                {
                    "timestamp": timestamp_hms,
                    "fully_visible": fully_visible_count,
                    "partially_visible": partially_visible_count,
                }
            )

        frame_no += 1
        # Update progress bar
        progress_percentage = min(frame_no / total_frames, 1.0) if total_frames > 0 else 0
        progress_bar.progress(progress_percentage, text=progress_text)

    cap.release()
    progress_bar.empty() # Clear the progress bar after completion
    logging.info("Video analysis completed.")

    return pd.DataFrame(results_list)

# ------------------------------------------------
# PLOTTING UTILITIES (for both Streamlit and PDF)
# ------------------------------------------------

def _create_pie_chart(data: list[float], labels: list[str], title: str, colors: list[str]) -> plt.Figure:
    """Helper to create a pie chart."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(data, labels=labels, autopct="%1.1f%%", startangle=90, colors=colors)
    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    return fig

def _create_second_wise_bar_chart(df: pd.DataFrame) -> plt.Figure:
    """Helper to create a second-wise bar chart."""
    x = np.arange(len(df))
    width = 0.35
    fig, ax = plt.subplots(figsize=(14, 6)) # Larger for better readability
    ax.bar(x - width/2, df["fully_visible"], width, label="Fully Visible", color=COLOR_FULLY_VISIBLE)
    ax.bar(x + width/2, df["partially_visible"], width, label="Partially Visible", color=COLOR_PARTIALLY_VISIBLE)
    ax.set_title("Second-wise Logo Visibility", fontsize=16)
    ax.set_xlabel("Timestamp (HH:MM:SS)", fontsize=12)
    ax.set_ylabel("Count of Logos Detected", fontsize=12)
    ax.legend()

    # Dynamic step for x-axis labels to prevent overcrowding
    step = max(1, len(df) // 20) # Show about 20 labels
    ax.set_xticks(x[::step])
    ax.set_xticklabels(df["timestamp"][::step], rotation=45, ha='right')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    return fig

def _create_minute_wise_bar_chart(df: pd.DataFrame) -> plt.Figure:
    """Helper to create a minute-wise bar chart."""
    df_temp = df.copy()
    df_temp["minute"] = df_temp.index // 60
    minute_data = df_temp.groupby("minute")[["fully_visible","partially_visible"]].sum()

    fig, ax = plt.subplots(figsize=(14, 6)) # Larger for better readability
    minute_data.plot(kind="bar", ax=ax, color=[COLOR_FULLY_VISIBLE, COLOR_PARTIALLY_VISIBLE])
    ax.set_title("Minute-wise Logo Visibility", fontsize=16)
    ax.set_xlabel("Minute", fontsize=12)
    ax.set_ylabel("Total Logos Detected", fontsize=12)
    ax.set_xticks(np.arange(len(minute_data)))
    ax.set_xticklabels(minute_data.index, rotation=0)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    return fig

# ------------------------------------------------
# PDF REPORT GENERATION
# ------------------------------------------------

def generate_pdf(
    df: pd.DataFrame,
    total_full: float,
    total_partial: float,
    not_visible: float,
    total_seconds: float,
    exposure_score: float
) -> None:
    """
    Generates a PDF report with analysis summaries and plots.

    Args:
        df (pd.DataFrame): The DataFrame containing second-wise visibility data.
        total_full (float): Total seconds the logo was fully visible.
        total_partial (float): Total seconds the logo was partially visible.
        not_visible (float): Total seconds the logo was not visible.
        total_seconds (float): Total duration of the video in seconds.
        exposure_score (float): The calculated advertisement exposure score.
    """
    logging.info("Generating PDF report...")

    # Ensure plots are generated with a clean state
    plt.close('all')

    # Generate and save plots
    fig1 = _create_pie_chart(
        [total_full, total_partial],
        ["Fully Visible", "Partially Visible"],
        "Full vs Partial Visibility",
        [COLOR_FULLY_VISIBLE, COLOR_PARTIALLY_VISIBLE]
    )
    fig1.savefig(TEMP_PLOT_FILES[0], bbox_inches='tight')
    plt.close(fig1)

    fig2 = _create_pie_chart(
        [total_full, total_partial, not_visible],
        ["Fully Visible", "Partially Visible", "Not Visible"],
        "Overall Visibility Distribution",
        [COLOR_FULLY_VISIBLE, COLOR_PARTIALLY_VISIBLE, COLOR_NOT_VISIBLE]
    )
    fig2.savefig(TEMP_PLOT_FILES[1], bbox_inches='tight')
    plt.close(fig2)

    fig3 = _create_second_wise_bar_chart(df)
    fig3.savefig(TEMP_PLOT_FILES[2], bbox_inches='tight')
    plt.close(fig3)

    fig4 = _create_minute_wise_bar_chart(df)
    fig4.savefig(TEMP_PLOT_FILES[3], bbox_inches='tight')
    plt.close(fig4)

    # Initialize PDF
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # --- Page 1: Title & Summary Metrics ---
    pdf.add_page()
    pdf.set_font("Arial", "B", 24)
    pdf.cell(0, 20, "Paytm Advertisement Visibility Report", ln=True, align='C')
    pdf.ln(10)

    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 8, f"Video Length: {seconds_to_hms(total_seconds)}", ln=True)
    pdf.cell(0, 8, f"Fully Visible Time: {seconds_to_hms(total_full)}", ln=True)
    pdf.cell(0, 8, f"Partially Visible Time: {seconds_to_hms(total_partial)}", ln=True)
    pdf.cell(0, 8, f"No Visibility Time: {seconds_to_hms(not_visible)}", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, f"Advertisement Exposure Score: {round(exposure_score, 2)}/100", ln=True)
    pdf.ln(10)

    # --- Page 2: Full vs Partial Visibility Pie Chart ---
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Full vs Partial Visibility", ln=True, align='C')
    x_pie1 = (PAGE_WIDTH_MM - IMAGE_WIDTH_PIE_MM) / 2
    y_pie1 = pdf.get_y() + MARGIN_BELOW_TITLE_MM
    pdf.image(TEMP_PLOT_FILES[0], x=x_pie1, y=y_pie1, w=IMAGE_WIDTH_PIE_MM)

    # --- Page 3: Overall Visibility Distribution Pie Chart (Centered) ---
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Overall Visibility Distribution", ln=True, align='C')
    x_pie2 = (PAGE_WIDTH_MM - IMAGE_WIDTH_PIE_MM) / 2
    y_pie2 = pdf.get_y() + MARGIN_BELOW_TITLE_MM
    pdf.image(TEMP_PLOT_FILES[1], x=x_pie2, y=y_pie2, w=IMAGE_WIDTH_PIE_MM)

    # --- Page 4: Second-wise Logo Visibility Graph ---
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Second-wise Logo Visibility", ln=True, align='C')
    x_bar1 = (PAGE_WIDTH_MM - IMAGE_WIDTH_BAR_MM) / 2
    y_bar1 = pdf.get_y() + MARGIN_BELOW_TITLE_MM
    pdf.image(TEMP_PLOT_FILES[2], x=x_bar1, y=y_bar1, w=IMAGE_WIDTH_BAR_MM)

    # --- Page 5: Minute-wise Logo Visibility Graph ---
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Minute-wise Logo Visibility", ln=True, align='C')
    x_bar2 = (PAGE_WIDTH_MM - IMAGE_WIDTH_BAR_MM) / 2
    y_bar2 = pdf.get_y() + MARGIN_BELOW_TITLE_MM
    pdf.image(TEMP_PLOT_FILES[3], x=x_bar2, y=y_bar2, w=IMAGE_WIDTH_BAR_MM)

    # --- Page 6+: Advertisement Visibility Timestamps Table ---
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Advertisement Visibility Timestamps", ln=True, align='C')
    pdf.ln(5) # Small gap before table headers

    pdf.set_font("Arial", "B", 11)
    pdf.cell(PDF_TABLE_COL_WIDTH_MM, 10, "Timestamp", 1, 0, 'C')
    pdf.cell(PDF_TABLE_COL_WIDTH_MM, 10, "Fully Visible", 1, 0, 'C')
    pdf.cell(PDF_TABLE_COL_WIDTH_MM, 10, "Partially Visible", 1, 1, 'C')

    pdf.set_font("Arial", "", 10)
    highlights = df[(df["fully_visible"] > 0) | (df["partially_visible"] > 0)]
    if highlights.empty:
        pdf.cell(0, 10, "No logos detected with visibility.", 1, 1, 'C')
    else:
        for _, row in highlights.iterrows():
            pdf.cell(PDF_TABLE_COL_WIDTH_MM, 8, str(row["timestamp"]), 1, 0, 'C')
            pdf.cell(PDF_TABLE_COL_WIDTH_MM, 8, str(row["fully_visible"]), 1, 0, 'C')
            pdf.cell(PDF_TABLE_COL_WIDTH_MM, 8, str(row["partially_visible"]), 1, 1, 'C')
            # Add new page if table gets too long
            if pdf.get_y() > (pdf.h - 30):
                pdf.add_page()
                pdf.set_font("Arial", "B", 11)
                pdf.cell(PDF_TABLE_COL_WIDTH_MM, 10, "Timestamp", 1, 0, 'C')
                pdf.cell(PDF_TABLE_COL_WIDTH_MM, 10, "Fully Visible", 1, 0, 'C')
                pdf.cell(PDF_TABLE_COL_WIDTH_MM, 10, "Partially Visible", 1, 1, 'C')
                pdf.set_font("Arial", "", 10)

    pdf.output(TEMP_PDF_REPORT)
    logging.info(f"PDF report generated as {TEMP_PDF_REPORT}")

    # Clean up generated plot images
    for img_file in TEMP_PLOT_FILES:
        if os.path.exists(img_file):
            os.remove(img_file)
            logging.debug(f"Cleaned up temporary plot file: {img_file}")


# ------------------------------------------------
# STREAMLIT UI
# ------------------------------------------------

st.markdown(
    f"""
    <h1 style='
        text-align: center;
        color: {COLOR_ACCENT_PRIMARY};
        font-size: 3.5em; /* Larger font size */
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2); /* Subtle shadow */
        margin-bottom: 0.5em; /* Space below title */
    '>
         Paytm Logo Visibility Analyzer
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    This AI-powered system analyzes video footage to detect and quantify **Paytm advertisement visibility**
    during broadcasts, such as cricket matches. Upload your video to get a detailed report on logo exposure.
    """
)

st.info("Upload a video file (MP4, AVI, MOV) to begin the analysis. The analysis processes one frame per second.")

uploaded_file = st.file_uploader(
    "Choose a video file",
    type=["mp4", "avi", "mov"]
)

# Ensure temporary video file is cleaned up reliably
if uploaded_file:
    # Use a try-finally block to ensure cleanup even if errors occur during analysis
    try:
        # Save the uploaded file temporarily
        with open(TEMP_VIDEO_FILE, "wb") as f:
            f.write(uploaded_file.read())
        logging.info(f"Uploaded file saved temporarily as {TEMP_VIDEO_FILE}")

        st.video(uploaded_file)
        st.markdown("---") # Visual separator

        if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
            with st.spinner("Analyzing video... This might take a while depending on video length and your system's performance."):
                df = analyze_video(TEMP_VIDEO_FILE)

            if not df.empty:
                st.success("Analysis Completed! See the report below.")
                logging.info("Video analysis successfully completed and data generated.")

                # Calculate summary statistics
                total_full = df["fully_visible"].sum()
                total_partial = df["partially_visible"].sum()
                not_visible = (
                    (df["fully_visible"] == 0) &
                    (df["partially_visible"] == 0)
                ).sum()
                total_seconds = len(df)

                # Avoid division by zero if video is too short or no frames processed
                if total_seconds > 0:
                    # Exposure score: Fully visible counts 100%, partially visible counts 50%
                    exposure_score = ((total_full * 100) + (total_partial * 50)) / total_seconds
                else:
                    exposure_score = 0.0

                st.header("📊 Analysis Summary")
                col1, col2, col3, col4, col5 = st.columns(5)

                col1.metric("Video Length", seconds_to_hms(total_seconds))
                col2.metric("Fully Visible Time", seconds_to_hms(total_full))
                col3.metric("Partially Visible Time", seconds_to_hms(total_partial))
                col4.metric("No Visibility Time", seconds_to_hms(not_visible))
                col5.metric("Exposure Score", f"{exposure_score:.1f}/100")

                st.divider()

                st.subheader("Visibility Distribution")
                col_pie1, col_pie2 = st.columns(2)

                # PIE CHART 1: Full vs Partial Visibility
                fig1 = _create_pie_chart(
                    [total_full, total_partial],
                    ["Fully Visible", "Partially Visible"],
                    "Full vs Partial Visibility",
                    [COLOR_FULLY_VISIBLE, COLOR_PARTIALLY_VISIBLE]
                )
                with col_pie1:
                    st.pyplot(fig1, use_container_width=True)
                plt.close(fig1) # Close plot to free memory

                # PIE CHART 2: Overall Visibility Distribution
                fig2 = _create_pie_chart(
                    [total_full, total_partial, not_visible],
                    ["Fully Visible", "Partially Visible", "Not Visible"],
                    "Overall Visibility Distribution",
                    [COLOR_FULLY_VISIBLE, COLOR_PARTIALLY_VISIBLE, COLOR_NOT_VISIBLE]
                )
                with col_pie2:
                    st.pyplot(fig2, use_container_width=True)
                plt.close(fig2) # Close plot to free memory

                st.divider()

                # SECOND-WISE GRAPH
                st.subheader("📈 Second-wise Logo Visibility Over Time")
                fig3 = _create_second_wise_bar_chart(df)
                st.pyplot(fig3, use_container_width=True)
                plt.close(fig3) # Close plot to free memory

                st.divider()

                # MINUTE-WISE GRAPH
                st.subheader("📊 Minute-wise Logo Visibility")
                fig4 = _create_minute_wise_bar_chart(df)
                st.pyplot(fig4, use_container_width=True)
                plt.close(fig4) # Close plot to free memory

                st.divider()

                # HIGHLIGHTS TABLE
                st.subheader("🔍 Advertisement Visibility Timestamps (Detected Logos)")
                highlights = df[
                    (df["fully_visible"] > 0) | (df["partially_visible"] > 0)
                ]
                if not highlights.empty:
                    st.dataframe(highlights, use_container_width=True)
                else:
                    st.info("No logos were detected with visibility in this video segment.")

                # FULL TIMESTAMP DATASET (in an expander)
                with st.expander("View Full Timestamp Dataset"):
                    st.dataframe(df, use_container_width=True)

                st.divider()

                # DOWNLOAD REPORTS
                st.header("⬇️ Download Reports")
                col_csv, col_pdf = st.columns(2)

                # CSV DOWNLOAD
                csv = df.to_csv(index=False).encode('utf-8')
                with col_csv:
                    st.download_button(
                        "Download CSV Report",
                        csv,
                        "paytm_logo_visibility_report.csv",
                        "text/csv",
                        key='download_csv',
                        use_container_width=True
                    )

                # PDF REPORT
                generate_pdf(
                    df,
                    total_full,
                    total_partial,
                    not_visible,
                    total_seconds,
                    exposure_score
                )
                with col_pdf:
                    if os.path.exists(TEMP_PDF_REPORT):
                        with open(TEMP_PDF_REPORT, "rb") as f:
                            st.download_button(
                                "Download PDF Report",
                                f.read(),
                                "paytm_visibility_report.pdf",
                                "application/pdf",
                                key='download_pdf',
                                use_container_width=True
                            )
                        os.remove(TEMP_PDF_REPORT) # Clean up PDF after download
                        logging.debug(f"Cleaned up temporary PDF file: {TEMP_PDF_REPORT}")
                    else:
                        st.error("PDF report could not be generated or found.")
                        logging.error(f"PDF report file not found for download: {TEMP_PDF_REPORT}")

            else:
                st.error("Analysis failed or no data was generated. Please check the video file and try again.")
                logging.warning("Analysis returned an empty DataFrame.")

    except Exception as e:
        st.error(f"An unexpected error occurred during video processing: {e}")
        logging.exception("An unexpected error occurred in the main Streamlit block.")
    finally:
        # Ensure the temporary video file is always removed
        if os.path.exists(TEMP_VIDEO_FILE):
            os.remove(TEMP_VIDEO_FILE)
            logging.info(f"Cleaned up temporary video file: {TEMP_VIDEO_FILE}")
else:
    # If no file is uploaded, ensure any leftover temp file is removed from previous runs
    if os.path.exists(TEMP_VIDEO_FILE):
        os.remove(TEMP_VIDEO_FILE)
        logging.info(f"Cleaned up stale temporary video file: {TEMP_VIDEO_FILE}")
    # Also clean up any leftover PDF report
    if os.path.exists(TEMP_PDF_REPORT):
        os.remove(TEMP_PDF_REPORT)
        logging.info(f"Cleaned up stale temporary PDF file: {TEMP_PDF_REPORT}")
    # And plot files
    for img_file in TEMP_PLOT_FILES:
        if os.path.exists(img_file):
            os.remove(img_file)
            logging.info(f"Cleaned up stale temporary plot file: {img_file}")