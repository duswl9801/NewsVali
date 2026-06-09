import os
import re
from pathlib import Path

import customtkinter as ctk
import joblib

from utils.configuration import Configuration

config = Configuration.from_file("config.json")
model_dir = config.get_str("app.model_dir", "./outputs/models")
model_name = "rf_grid_best_model_grid_search.pkl"
MODEL = os.path.join(model_dir, model_name)

VECTORIZER = "dataset/tfidf_vectorizer.pkl"

class NewsVerificationApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("I-PIX | News Verification Assistant")
        self.geometry("1320x860")
        self.minsize(1120, 720)

        ctk.set_appearance_mode("dark")
        self.current_theme_name = "blue_night"

        self.themes = {
            "blue_night": {
                "bg": "#0f1117",
                "card": "#171a21",
                "card_2": "#11151c",
                "border": "#2a2f3a",
                "text": "#f3f5f7",
                "soft_text": "#aeb6c3",
                "accent": "#6ea8fe",
                "accent_hover": "#5b96f5",
                "selected": "#1d2d47",
                "selected_border": "#6ea8fe",
                "tag_needs": "#d65c6a",
                "tag_suspicious": "#d4b14a",
                "tag_reliable": "#7c8797",
                "hl_needs_bg": "#5a242d",
                "hl_suspicious_bg": "#5a4c1f",
                "hl_needs_fg": "#fff2f4",
                "hl_suspicious_fg": "#fff8df",
            },
            "forest_ink": {
                "bg": "#0f1412",
                "card": "#18201c",
                "card_2": "#111714",
                "border": "#2d3a34",
                "text": "#eef5ef",
                "soft_text": "#a9b9b0",
                "accent": "#7bc6a4",
                "accent_hover": "#68b490",
                "selected": "#1d3128",
                "selected_border": "#7bc6a4",
                "tag_needs": "#de6975",
                "tag_suspicious": "#d2ba56",
                "tag_reliable": "#8da295",
                "hl_needs_bg": "#5a2a30",
                "hl_suspicious_bg": "#5f5524",
                "hl_needs_fg": "#fff3f4",
                "hl_suspicious_fg": "#fff8df",
            },
            "plum_graphite": {
                "bg": "#120f15",
                "card": "#1c1722",
                "card_2": "#15111a",
                "border": "#372d41",
                "text": "#f5f2f7",
                "soft_text": "#b9afc3",
                "accent": "#c793ff",
                "accent_hover": "#b77df5",
                "selected": "#312142",
                "selected_border": "#c793ff",
                "tag_needs": "#e07184",
                "tag_suspicious": "#ddc15c",
                "tag_reliable": "#9387a5",
                "hl_needs_bg": "#612933",
                "hl_suspicious_bg": "#665623",
                "hl_needs_fg": "#fff3f6",
                "hl_suspicious_fg": "#fff8de",
            },
        }

        self._apply_theme_values()

        self.demo_url = "https://demo-news.example/article/sample-story"
        self.sentences = [
            "Government officials confirmed that the new law reduced unemployment by 30% in just two weeks.",
            "Many residents said they felt hopeful after the announcement.",
            "A recent study proved that the policy worked better than any previous reform in the country.",
            "According to local analysts, the reported figures may require further review before any final conclusion.",
            "The city library will remain open until 8 p.m. this weekend.",
        ]

        self.sentence_results = [
            {
                "label": "needs verification",
                "probability": "0.88",
                "why_flagged": (
                    "• Contains a numerical/statistical claim\n"
                    "• References government officials as an authority\n"
                    "• Makes a strong real-world policy outcome claim\n"
                    "• Uses assertive wording such as 'confirmed'"
                ),
            },
            {
                "label": "likely reliable",
                "probability": "0.21",
                "why_flagged": (
                    "• Mostly expresses public reaction rather than a strong factual claim\n"
                    "• No numerical or high-risk claim detected"
                ),
            },
            {
                "label": "suspicious",
                "probability": "0.61",
                "why_flagged": (
                    "• References a study without concrete details\n"
                    "• Uses highly assertive wording such as 'proved'\n"
                    "• Makes a broad comparative claim"
                ),
            },
            {
                "label": "needs verification",
                "probability": "0.79",
                "why_flagged": (
                    "• Refers to reported figures that require validation\n"
                    "• Mentions analysts and an unresolved claim\n"
                    "• Suggests uncertainty and need for follow-up checking"
                ),
            },
            {
                "label": "likely reliable",
                "probability": "0.16",
                "why_flagged": (
                    "• Low-risk logistical information\n"
                    "• Does not contain a strong controversial or quantitative claim"
                ),
            },
        ]

        self.model, self.vectorizer = self._load_model_artifacts(MODEL, VECTORIZER)

        self.selected_sentence_index = 0
        self.sentence_ranges = []  # stores start/end indices in textbox
        self.highlight_tag_map = {}  # tag_name -> sentence index

        self._build_layout()
        self.load_demo_content()
        self.apply_highlights_to_article()
        self.select_sentence(0)

    def _apply_theme_values(self):
        theme = self.themes[self.current_theme_name]
        self.bg_color = theme["bg"]
        self.card_color = theme["card"]
        self.card_color_2 = theme["card_2"]
        self.card_border = theme["border"]
        self.main_text = theme["text"]
        self.soft_text = theme["soft_text"]
        self.accent = theme["accent"]
        self.accent_hover = theme["accent_hover"]
        self.selected_fill = theme["selected"]
        self.selected_border = theme["selected_border"]

        self.tag_needs = theme["tag_needs"]
        self.tag_suspicious = theme["tag_suspicious"]
        self.tag_reliable = theme["tag_reliable"]

        self.hl_needs_bg = theme["hl_needs_bg"]
        self.hl_suspicious_bg = theme["hl_suspicious_bg"]
        self.hl_needs_fg = theme["hl_needs_fg"]
        self.hl_suspicious_fg = theme["hl_suspicious_fg"]

        self.configure(fg_color=self.bg_color)

    def _build_layout(self):
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(0, weight=1)

        self._build_left_panel()
        self._build_right_panel()
        self._build_bottom_row()

    def _build_left_panel(self):
        self.left_card = ctk.CTkFrame(
            self,
            fg_color=self.card_color,
            border_width=1,
            border_color=self.card_border,
            corner_radius=24,
        )
        self.left_card.grid(row=0, column=0, sticky="nsew", padx=(22, 12), pady=(20, 12))
        self.left_card.grid_columnconfigure(0, weight=1)
        self.left_card.grid_rowconfigure(3, weight=1)

        self.left_title = ctk.CTkLabel(
            self.left_card,
            text="Article Input",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color=self.main_text,
        )
        self.left_title.grid(row=0, column=0, sticky="w", padx=18, pady=(18, 4))

        self.left_subtitle = ctk.CTkLabel(
            self.left_card,
            text="Click highlighted sentences inside the article. Red = needs verification, Yellow = suspicious. Likely reliable sentences are not highlighted.",
            font=ctk.CTkFont(size=13),
            text_color=self.soft_text,
            wraplength=720,
            justify="left",
        )
        self.left_subtitle.grid(row=1, column=0, sticky="w", padx=18, pady=(0, 14))

        self.url_entry = ctk.CTkEntry(
            self.left_card,
            height=42,
            corner_radius=14,
            border_width=1,
            border_color=self.card_border,
            fg_color=self.card_color_2,
            text_color=self.main_text,
            font=ctk.CTkFont(size=14),
        )
        self.url_entry.grid(row=2, column=0, sticky="ew", padx=18, pady=(0, 14))

        self.input_box = ctk.CTkTextbox(
            self.left_card,
            corner_radius=18,
            border_width=1,
            border_color=self.card_border,
            fg_color=self.card_color_2,
            text_color=self.main_text,
            font=ctk.CTkFont(size=15),
            wrap="word",
        )
        self.input_box.grid(row=3, column=0, sticky="nsew", padx=18, pady=(0, 18))

    def _build_right_panel(self):
        self.right_card = ctk.CTkFrame(
            self,
            fg_color=self.card_color,
            border_width=1,
            border_color=self.card_border,
            corner_radius=24,
        )
        self.right_card.grid(row=0, column=1, sticky="nsew", padx=(12, 22), pady=(20, 12))
        self.right_card.grid_columnconfigure(0, weight=1)

        self.output_title = ctk.CTkLabel(
            self.right_card,
            text="Output",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color=self.main_text,
        )
        self.output_title.grid(row=0, column=0, sticky="w", padx=18, pady=(18, 6))

        self.output_subtitle = ctk.CTkLabel(
            self.right_card,
            text="Click a highlighted sentence in the article to inspect its result.",
            font=ctk.CTkFont(size=13),
            text_color=self.soft_text,
        )
        self.output_subtitle.grid(row=1, column=0, sticky="w", padx=18, pady=(0, 16))

        self.selected_sentence_box = self._make_info_box("Selected Sentence", 2, 150)
        self.selected_sentence_value = self.selected_sentence_box["value"]

        self.label_box = self._make_info_box("Label", 3, 86)
        self.label_value = self.label_box["value"]

        self.prob_box = self._make_info_box("Probability", 4, 86)
        self.prob_value = self.prob_box["value"]

        self.reason_box = self._make_info_box("Why Flagged", 5, 220)
        self.reason_value = self.reason_box["value"]

    def _make_info_box(self, title, row, min_height=100):
        frame = ctk.CTkFrame(
            self.right_card,
            fg_color=self.card_color_2,
            border_width=1,
            border_color=self.card_border,
            corner_radius=18,
            height=min_height,
        )
        frame.grid(row=row, column=0, sticky="ew", padx=18, pady=(0, 14))
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_propagate(False)

        title_label = ctk.CTkLabel(
            frame,
            text=title,
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=self.soft_text,
        )
        title_label.grid(row=0, column=0, sticky="w", padx=16, pady=(12, 4))

        value_label = ctk.CTkLabel(
            frame,
            text="-",
            font=ctk.CTkFont(size=16),
            text_color=self.main_text,
            justify="left",
            anchor="nw",
            wraplength=390,
        )
        value_label.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 12))

        return {"frame": frame, "title": title_label, "value": value_label}

    def _build_bottom_row(self):
        self.bottom_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.bottom_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=22, pady=(0, 20))
        self.bottom_frame.grid_columnconfigure(0, weight=0)
        self.bottom_frame.grid_columnconfigure(1, weight=0)
        self.bottom_frame.grid_columnconfigure(2, weight=0)
        self.bottom_frame.grid_columnconfigure(3, weight=1)
        self.bottom_frame.grid_columnconfigure(4, weight=0)
        self.bottom_frame.grid_columnconfigure(5, weight=0)

        self.load_demo_button = ctk.CTkButton(
            self.bottom_frame,
            text="Load Demo Article",
            height=46,
            corner_radius=14,
            fg_color=self.accent,
            hover_color=self.accent_hover,
            text_color="#0b1220",
            font=ctk.CTkFont(size=15, weight="bold"),
            command=self._reload_demo_and_highlight,
        )
        self.load_demo_button.grid(row=0, column=0, sticky="w")

        self.theme_label = ctk.CTkLabel(
            self.bottom_frame,
            text="Theme",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=self.soft_text,
        )
        self.theme_label.grid(row=0, column=4, sticky="e", padx=(0, 8))

        self.run_model_button = ctk.CTkButton(
            self.bottom_frame,
            text="Run Model Test",
            height=46,
            corner_radius=14,
            fg_color=self.accent,
            hover_color=self.accent_hover,
            text_color="#0b1220",
            font=ctk.CTkFont(size=15, weight="bold"),
            command=self.run_model_test,
        )
        self.run_model_button.grid(row=0, column=1, sticky="w", padx=(10, 0))

        self.clear_button = ctk.CTkButton(
            self.bottom_frame,
            text="Clear",
            height=46,
            corner_radius=14,
            fg_color=self.card_color,
            hover_color=self.selected_fill,
            text_color=self.main_text,
            font=ctk.CTkFont(size=15, weight="bold"),
            command=self.clear_article,
        )
        self.clear_button.grid(row=0, column=2, sticky="w", padx=(10, 0))

        self.theme_menu = ctk.CTkOptionMenu(
            self.bottom_frame,
            values=["blue_night", "forest_ink", "plum_graphite"],
            command=self.change_theme,
            width=180,
            height=38,
            corner_radius=12,
            fg_color=self.card_color,
            button_color=self.accent,
            button_hover_color=self.accent_hover,
            dropdown_fg_color=self.card_color,
            dropdown_hover_color=self.selected_fill,
            text_color=self.main_text,
            font=ctk.CTkFont(size=14),
        )
        self.theme_menu.set(self.current_theme_name)
        self.theme_menu.grid(row=0, column=5, sticky="e")

    def _load_model_artifacts(self, model_path, vectorizer):
        """
        Load a trained sklearn model.

        supported options:
        1. models/best_pipeline.pkl
           - a single sklearn Pipeline that already includes TF-IDF + classifier
        2. models/tfidf_vectorizer.pkl + models/best_model.pkl
           - separate vectorizer and classifier

        If no model file exists, the app still runs with a simple heuristic fallback
        so the UI can be tested before the final model is connected.
        """
        #pipeline_path = Path("models/best_pipeline.pkl")
        vectorizer = joblib.load(vectorizer)
        model = joblib.load(model_path)

        #if pipeline_path.exists():
        #    return joblib.load(pipeline_path), None

        return model, vectorizer


    def _split_into_sentences(self, article_text):
        """Split pasted article text into sentence-like units without extra dependencies."""
        clean_text = re.sub(r"\s+", " ", article_text).strip()
        if not clean_text:
            return []

        # handles most news sentences ending with ., ?, or !
        sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"“])", clean_text)
        return [sentence.strip() for sentence in sentences if len(sentence.strip()) > 5]

    def run_model_test(self):
        """Read pasted article text, run sentence-level prediction, and refresh highlights/output."""
        article_text = self.input_box.get("1.0", "end-1c").strip()
        sentences = self._split_into_sentences(article_text)

        if not sentences:
            self.sentences = []
            self.sentence_results = []
            self.apply_highlights_to_article()
            self.selected_sentence_value.configure(text="Paste an article first.")
            self.label_value.configure(text="-")
            self.prob_value.configure(text="-")
            self.reason_value.configure(text="-")
            return

        self.sentences = sentences
        self.sentence_results = [self._predict_sentence(sentence) for sentence in sentences]

        print("\n===== sentence prediction results =====")
        for i, (sentence, result) in enumerate(zip(self.sentences, self.sentence_results), start=1):
            print(f"[{i}] {result['label']} | score={result['probability']}")
            print(sentence)
            print()

        self.apply_highlights_to_article()
        self.select_sentence(0)

    def _predict_sentence(self, sentence):
        """Return one UI-ready result dictionary for a sentence."""
        probability = self._predict_need_verification_probability(sentence)

        if probability >= 0.80:
            label = "needs verification"
        elif probability >= 0.64:
            label = "suspicious"
        else:
            label = "likely reliable"

        return {
            "label": label,
            "probability": f"{probability:.2f}",
            "why_flagged": self._build_reason(sentence, probability, label),
        }

    def _predict_need_verification_probability(self, sentence):
        """Predict probability for the positive/check-worthy class."""
        if self.model is None:
            return self._heuristic_probability(sentence)

        X = [sentence]
        if self.vectorizer is not None:
            X = self.vectorizer.transform(X)

        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(X)[0]
            classes = list(getattr(self.model, "classes_", []))
            positive_index = self._positive_class_index(classes)
            return float(probabilities[positive_index])

        if hasattr(self.model, "decision_function"):
            score = float(self.model.decision_function(X)[0])
            return 1 / (1 + pow(2.718281828, -score))

        prediction = self.model.predict(X)[0]
        return 0.75 if str(prediction).lower() in {"1", "true", "needs verification", "check-worthy"} else 0.25

    def _positive_class_index(self, classes):
        """Find the index for the positive class used by the trained model."""
        positive_names = {"1", "true", "needs verification", "need verification", "check-worthy", "checkworthy"}
        for idx, class_name in enumerate(classes):
            if str(class_name).strip().lower() in positive_names:
                return idx
        return 1 if len(classes) > 1 else 0

    def _heuristic_probability(self, sentence):
        """Temporary fallback until the saved model files are connected."""
        s = sentence.lower()
        score = 0.15

        risky_words = [
            "confirmed", "proved", "study", "report", "according to", "officials",
            "bill", "senate", "house", "court", "lawsuit", "ipo", "billion",
            "million", "percent", "%", "rise", "increase", "decrease", "reduced",
        ]
        score += 0.10 * sum(word in s for word in risky_words)

        if re.search(r"\$?\d+(\.\d+)?\s?(%|percent|million|billion|trillion)?", sentence):
            score += 0.25

        if any(quote in sentence for quote in ['"', "“", "”"]):
            score += 0.08

        return min(score, 0.95)

    def _build_reason(self, sentence, probability, label):
        """Create a short explanation for the selected sentence."""
        reasons = []
        lower_sentence = sentence.lower()

        if re.search(r"\$?\d+(\.\d+)?\s?(%|percent|million|billion|trillion)?", sentence):
            reasons.append("• Contains a numerical or financial claim")
        if any(word in lower_sentence for word in ["study", "report", "according to", "officials", "analysts"]):
            reasons.append("• References a study, report, official, or analyst source")
        if any(word in lower_sentence for word in
               ["senate", "house", "bill", "court", "lawsuit", "government", "policy"]):
            reasons.append("• Mentions a political, legal, or policy-related claim")
        if any(word in lower_sentence for word in ["proved", "confirmed", "best", "worst", "always", "never"]):
            reasons.append("• Uses strong or assertive wording")

        if not reasons:
            if label == "likely reliable":
                reasons.append("• Low-risk sentence with no strong factual or numerical claim detected")
            else:
                reasons.append("• Model score suggests this sentence may need follow-up checking")

        reasons.append(f"• Model score: {probability:.2f}")
        return "\n".join(reasons)

    def clear_article(self):
        """Clear article input and output panel."""
        text_widget = self.input_box._textbox
        text_widget.config(state="normal")
        text_widget.delete("1.0", "end")

        self.sentences = []
        self.sentence_results = []
        self.sentence_ranges = []
        self.highlight_tag_map = {}
        self.selected_sentence_index = 0

        self.selected_sentence_value.configure(text="-")
        self.label_value.configure(text="-")
        self.prob_value.configure(text="-")
        self.reason_value.configure(text="-")

        text_widget.config(state="normal")

    def load_demo_content(self):
        self.url_entry.delete(0, "end")
        self.url_entry.insert(0, self.demo_url)

        article_text = " ".join(self.sentences)
        self.input_box.delete("1.0", "end")
        self.input_box.insert("1.0", article_text)

    def _reload_demo_and_highlight(self):
        self.load_demo_content()
        self.apply_highlights_to_article()
        self.select_sentence(0)

    def _display_tag(self, label):
        if label == "needs verification":
            return "[needs verification]"
        if label == "suspicious":
            return "[suspicious]"
        return "[likely reliable]"

    def apply_highlights_to_article(self):
        """
        Highlight only 'needs verification' and 'suspicious' directly inside the article textbox.
        No inline label text is inserted into the article.
        """
        text_widget = self.input_box._textbox  # underlying tkinter.Text widget
        text_widget.config(state="normal")

        text_widget.delete("1.0", "end")

        self.sentence_ranges = []
        self.highlight_tag_map = {}

        for idx, sentence in enumerate(self.sentences):
            result = self.sentence_results[idx]
            label = result["label"]

            start_index = text_widget.index("end-1c")
            if start_index == "1.0" and text_widget.get("1.0", "end-1c") == "":
                start_index = "1.0"

            text_widget.insert("end", sentence)
            end_index = text_widget.index("end-1c")

            self.sentence_ranges.append((start_index, end_index))

            if label in ["needs verification", "suspicious"]:
                sentence_tag = f"sentence_{idx}"
                self.highlight_tag_map[sentence_tag] = idx

                if label == "needs verification":
                    text_widget.tag_config(
                        sentence_tag,
                        background=self.hl_needs_bg,
                        foreground=self.hl_needs_fg,
                    )
                else:
                    text_widget.tag_config(
                        sentence_tag,
                        background=self.hl_suspicious_bg,
                        foreground=self.hl_suspicious_fg,
                    )

                text_widget.tag_add(sentence_tag, start_index, end_index)
                text_widget.tag_bind(sentence_tag, "<Button-1>", lambda e, i=idx: self.select_sentence(i))
                text_widget.tag_bind(sentence_tag, "<Enter>", lambda e: text_widget.config(cursor="hand2"))
                text_widget.tag_bind(sentence_tag, "<Leave>", lambda e: text_widget.config(cursor="xterm"))

            text_widget.insert("end", " ")

        text_widget.config(state="normal")

    def select_sentence(self, index):
        self.selected_sentence_index = index

        sentence = self.sentences[index]
        result = self.sentence_results[index]

        self.selected_sentence_value.configure(text=sentence)
        self.label_value.configure(text=result["label"])
        self.prob_value.configure(text=result["probability"])
        self.reason_value.configure(text=result["why_flagged"])

    def change_theme(self, theme_name):
        self.current_theme_name = theme_name
        self._apply_theme_values()
        self._refresh_theme()

    def _refresh_theme(self):
        self.configure(fg_color=self.bg_color)

        self.left_card.configure(fg_color=self.card_color, border_color=self.card_border)
        self.left_title.configure(text_color=self.main_text)
        self.left_subtitle.configure(text_color=self.soft_text)
        self.url_entry.configure(
            fg_color=self.card_color_2,
            border_color=self.card_border,
            text_color=self.main_text,
        )
        self.input_box.configure(
            fg_color=self.card_color_2,
            border_color=self.card_border,
            text_color=self.main_text,
        )

        self.right_card.configure(fg_color=self.card_color, border_color=self.card_border)
        self.output_title.configure(text_color=self.main_text)
        self.output_subtitle.configure(text_color=self.soft_text)

        for box in [self.selected_sentence_box, self.label_box, self.prob_box, self.reason_box]:
            box["frame"].configure(fg_color=self.card_color_2, border_color=self.card_border)
            box["title"].configure(text_color=self.soft_text)
            box["value"].configure(text_color=self.main_text)

        self.load_demo_button.configure(fg_color=self.accent, hover_color=self.accent_hover)
        self.theme_label.configure(text_color=self.soft_text)
        self.theme_menu.configure(
            fg_color=self.card_color,
            button_color=self.accent,
            button_hover_color=self.accent_hover,
            dropdown_fg_color=self.card_color,
            dropdown_hover_color=self.selected_fill,
            text_color=self.main_text,
        )

        self.apply_highlights_to_article()
        self.select_sentence(self.selected_sentence_index)


if __name__ == "__main__":
    app = NewsVerificationApp()
    app.mainloop()