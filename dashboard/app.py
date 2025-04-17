import os

os.environ["NUMBA_CACHE_DIR"] = "/tmp"
import polars as pl
import ufal.morphodita
from bertopic import BERTopic
from bertopic.dimensionality import BaseDimensionalityReduction
from bertopic.representation import KeyBERTInspired, LiteLLM, MaximalMarginalRelevance
from bertopic.vectorizers import ClassTfidfTransformer

# Load data and compute static values
from shared import (
    PROMPT,
    app_dir,
    embedding_model,
    questions_df,
    rc_dict,
    stopwords,
    tagger,
)
from shiny import App, Session, reactive, render, ui
from shinywidgets import output_widget, render_plotly
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

# Add page title and sidebar
app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.input_selectize(
            "resource_set",
            "Select Resource Set:",
            rc_dict,
        ),
        ui.input_slider("top_n_words", "Top n words", min=2, max=20, value=8),
        ui.input_slider("min_topic_size", "Min Topic Size", min=2, max=30, value=5),
        ui.input_switch("use_embeddings", "Visualize Embeddings", True),
        ui.input_switch("use_UMAP", "use UMAP", True),
        # ui.input_switch("use_model", "Use Model labelling", False),
        ui.input_action_button("run_pipeline", "Run Clustering"),
        open="desktop",
    ),
    ui.navset_pill(
        ui.nav_panel(
            "Documents Topic Plot",
            ui.card(
                ui.card_header(
                    "Topics",
                ),
                output_widget("topic_docs_plot"),
                full_screen=True,
            ),
        ),
        ui.nav_panel(
            "Documents Topic Table",
            ui.card(
                ui.card_header("Documents Topics table"),
                ui.output_data_frame("cluster_table"),
                full_screen=True,
            ),
        ),
        ui.nav_panel(
            "DataMap",
            ui.card(
                ui.card_header(
                    "DataMap",
                ),
                ui.output_ui("datamap"),
                full_screen=True,
            ),
        ),
        ui.nav_panel(
            "Topic Similarity",
            ui.layout_columns(
                ui.card(
                    ui.card_header("Intertopic plot"),
                    output_widget("intertopic_plot"),
                    full_screen=True,
                ),
                ui.card(
                    ui.card_header("Similarity matrix"),
                    ui.card_body(
                        output_widget("similarity_matrix"),
                    ),
                    full_screen=True,
                ),
            ),
        ),
        ui.nav_panel(
            "Topic Dendrogram",
            ui.card(
                ui.card_header(
                    "Topic hierarchy",
                ),
                output_widget("dendrogram"),
                full_screen=True,
            ),
        ),
        ui.nav_panel(
            "Topic Word Scores",
            ui.card(
                ui.card_header(
                    "Topic Word Scores",
                ),
                output_widget("topic_word_scores"),
                full_screen=True,
            ),
        ),
        ui.nav_panel(
            "Data Table",
            ui.card(
                ui.card_header("Tips data"),
                ui.output_data_frame("table"),
                full_screen=True,
            ),
        ),
        id="tab",
    ),
    ui.include_css(app_dir / "styles.css"),
    title="Rozhodovacka clustering",
    fillable=True,
)


class LemmaTokenizer:
    def __init__(self):
        self.tagger = tagger
        self.converter = ufal.morphodita.TagsetConverter.newStripLemmaIdConverter(
            self.tagger.getMorpho()
        )
        self.tokenizer = self.tagger.newTokenizer()
        self.forms = ufal.morphodita.Forms()

    def __call__(self, text):
        self.tokenizer.setText(text)
        self.tokenizer.nextSentence(self.forms, None)

        lemmas = ufal.morphodita.TaggedLemmas()
        self.tagger.tag(self.forms, lemmas)

        self.converter.convertAnalyzed(lemmas)

        raw_lemmas = list(set([lemma.lemma for lemma in lemmas]))
        return raw_lemmas


# Initialize CountVectorizer with the custom tokenizer
vectorizer = CountVectorizer(
    tokenizer=LemmaTokenizer(), strip_accents="unicode", lowercase=True
)


def server(input, output, session: Session):
    @reactive.calc
    def questions_data():
        rs_id = input.resource_set()
        return questions_df.filter((pl.col("rs") == int(rs_id)))

    @reactive.calc
    def topic_model():
        print("COMPUTING")

        vectorizer_model = CountVectorizer(
            stop_words=stopwords + ["img", "pravda", "nepravda"],
            tokenizer=LemmaTokenizer(),
            lowercase=True,
        )

        ctfidf_model = ClassTfidfTransformer(
            bm25_weighting=True, reduce_frequent_words=True
        )

        representation_models = [KeyBERTInspired(), MaximalMarginalRelevance(0.5)]

        reduction_model = BaseDimensionalityReduction()
        if input.use_UMAP():
            reduction_model = UMAP(
                n_neighbors=15,
                n_components=5,
                min_dist=0.0,
                metric="cosine",
                random_state=42,
            )

        if input.use_model():
            representation_model_LLM = LiteLLM(
                model="github/DeepSeek-V3", prompt=PROMPT, nr_docs=4
            )
            representation_models.append(representation_model_LLM)

        topic_model = BERTopic(
            embedding_model=embedding_model,
            ctfidf_model=ctfidf_model,
            umap_model=reduction_model,
            representation_model=representation_models,
            vectorizer_model=vectorizer_model,
            verbose=True,
            n_gram_range=(1, 4),
            top_n_words=input.top_n_words(),
            min_topic_size=input.min_topic_size(),
            language="multilingual",
        ).fit(docs(), embeddings=docs_embeddings())
        return topic_model

    @reactive.calc
    def docs_embeddings():
        return embedding_model.encode(docs())

    @reactive.calc
    def reduced_embeddings():
        from umap import UMAP

        return UMAP(
            n_neighbors=10, n_components=2, min_dist=0.0, metric="cosine"
        ).fit_transform(docs_embeddings())

    @reactive.calc
    def docs():
        return questions_data()["question_correct"].to_numpy().flatten().tolist()

    @reactive.calc
    def hierarchical_topics():
        from scipy.cluster import hierarchy as sch

        def linkage_function(x):
            return sch.linkage(x, "average", optimal_ordering=True)

        hierarchical_topics = topic_model().hierarchical_topics(
            docs(), linkage_function=linkage_function
        )
        return hierarchical_topics

    @render_plotly
    @reactive.event(input.run_pipeline, ignore_none=False)
    def topic_word_scores():
        return topic_model().visualize_barchart(height=500)

    @render_plotly
    @reactive.event(input.run_pipeline, ignore_none=False)
    def intertopic_plot():
        return topic_model().visualize_topics()

    @render_plotly
    @reactive.event(input.run_pipeline, ignore_none=False)
    def dendrogram():
        return topic_model().visualize_hierarchy(
            hierarchical_topics=hierarchical_topics()
        )

    @render_plotly
    @reactive.event(input.run_pipeline, ignore_none=False)
    def similarity_matrix():
        return topic_model().visualize_heatmap()

    @render.ui  # Correct decorator
    @reactive.event(input.run_pipeline, ignore_none=False)
    def datamap():
        import urllib.parse

        import datamapplot

        print("Attempting to render datamap...")
        try:
            # --- Ensure inputs are ready ---
            current_reduced_embeddings = reduced_embeddings()
            current_docs = docs()
            doc_info = topic_model().get_document_info(current_docs)
            topic_labels = doc_info["Name"].to_list()

            if not (
                current_reduced_embeddings.shape[0]
                == len(topic_labels)
                == len(current_docs)
            ):
                raise ValueError(
                    "Mismatch between embeddings, labels, and hover text counts."
                )

            # --- Plot Generation ---
            plot_object = datamapplot.create_interactive_plot(
                current_reduced_embeddings,
                topic_labels,
                font_family="Playfair Display SC",
                hover_text=current_docs,
                cluster_boundary_polygons=True,
                cluster_boundary_line_width=6,
                title="Rozhodovacka - clustering based on topics",
                polygon_alpha=0.5,
                enable_search=True,
            )

            # --- Get HTML Content using str() ---
            html_content = str(plot_object)

            if not html_content or not html_content.strip().startswith("<"):
                raise ValueError("Generated HTML content seems invalid or empty.")

            encoded_html = urllib.parse.quote(html_content)
            data_uri = f"data:text/html;charset=utf-8,{encoded_html}"

            return ui.tags.iframe(
                src=data_uri,
                width="100%",
                height="800px",  # Adjust height as needed
                style="border: none;",
            )

        except Exception as e:
            error_message = f"Error rendering datamap: {e}"
            print(error_message)
            return ui.HTML()  # Show traceback in UI too

    @render_plotly
    @reactive.event(input.run_pipeline, ignore_none=False)
    def topic_docs_plot():
        if input.use_embeddings():
            fig = topic_model().visualize_documents(
                docs(),
                embeddings=docs_embeddings(),
                reduced_embeddings=reduced_embeddings(),
            )
        else:
            fig = topic_model().visualize_documents(docs())

        marker_size = 10
        fig.update_traces(
            marker=dict(size=marker_size), selector=dict(type="scattergl")
        )
        cluster_label_font_size = 18

        fig.update_traces(
            textfont=dict(size=cluster_label_font_size),
            selector=dict(mode="markers+text"),
        )

        # Increase font sizes (adjust values as needed)
        title_font_size = 20
        axis_title_font_size = 16
        tick_font_size = 16
        legend_font_size = 14
        hover_font_size = 16

        resource_set_id = int(input.resource_set())
        fig.update_layout(
            title=rc_dict[resource_set_id] + " Documents and Topics",
            title_font_size=title_font_size,
            xaxis_title_font_size=axis_title_font_size,
            yaxis_title_font_size=axis_title_font_size,
            xaxis_tickfont_size=tick_font_size,
            yaxis_tickfont_size=tick_font_size,
            legend_title_font_size=legend_font_size
            + 1,  # Slightly larger title for legend
            legend_font_size=legend_font_size,  # Font size for legend items themselves
            hoverlabel_font_size=hover_font_size,  # Font size for the hover box
        )

        return fig

    @render.data_frame
    def cluster_table():
        topics_df = pl.from_pandas(topic_model().get_document_info(docs()))
        return render.DataGrid(topics_df, height="90vh", filters=True)

    @render.data_frame
    def table():
        return render.DataGrid(questions_data())

    @reactive.effect
    @reactive.event(input.reset)
    def _(): ...


app = App(app_ui, server)
