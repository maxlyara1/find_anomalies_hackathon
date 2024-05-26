import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import plotly.express as px
import plotly.graph_objs as go
import shap
from statsmodels.tsa.seasonal import seasonal_decompose
import streamlit.components.v1 as components

class FileProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.time_col = 'point'
        self.df = None

    def load_data(self):
        column_names = [
            'account_id', 'name', self.time_col, 'call_count', 'total_call_time', 
            'total_exclusive_time', 'min_call_time', 'max_call_time', 'sum_of_squares', 
            'instances', 'language', 'app_name', 'app_id', 'scope', 'host', 
            'display_host', 'pid', 'agent_version', 'labels'
        ]
        self.df = pd.read_csv(self.file_path, sep='\t', names=column_names)
        self.df[self.time_col] = pd.to_datetime(self.df[self.time_col])
        
    def calculate_throughput(self):
        Throughput_df = self.df[
            (self.df['language'] == 'java') &
            (self.df['app_name'] == '[GMonit] Collector') &
            (self.df['scope'].isna()) &
            (self.df['name'] == 'HttpDispatcher')
        ]
        aggregated_Throughput_df = Throughput_df.groupby(self.time_col, group_keys=False)['call_count'].sum().reset_index()
        aggregated_Throughput_df.columns = ['point', 'sum_call_count']
        
        # Remove seasonality
        decomposition = seasonal_decompose(aggregated_Throughput_df['sum_call_count'], period=1440, model='additive')
        aggregated_Throughput_df['seasonally_adjusted_sum_calls'] = aggregated_Throughput_df['sum_call_count'] - decomposition.seasonal
        aggregated_Throughput_df['seasonality'] = decomposition.seasonal

        return aggregated_Throughput_df
    
    def calculate_web_response(self):
        WebResponse_df = self.df[
            (self.df['language'] == 'java') &
            (self.df['app_name'] == '[GMonit] Collector') &
            (self.df['scope'].isna()) &
            (self.df['name'] == 'HttpDispatcher')
        ]
        WebResponse_df = WebResponse_df.groupby(self.time_col, group_keys=False).apply(
            lambda x: (x['total_call_time'].sum() / x['call_count'].sum()), include_groups=False
        ).reset_index()
        WebResponse_df.columns = ['point', 'web_response']
        return WebResponse_df
    
    def calculate_apdex(self):
        Apdex_df = self.df[
            (self.df['language'] == 'java') &
            (self.df['app_name'] == '[GMonit] Collector') &
            (self.df['scope'].isna()) &
            (self.df['name'] == 'Apdex')
        ]
        Apdex_df = Apdex_df.groupby(self.time_col, group_keys=False).apply(
            lambda x: (x['call_count'].sum() + x['total_call_time'].sum() / 2) / 
                      (x['call_count'].sum() + x['total_call_time'].sum() + x['total_exclusive_time'].sum()), include_groups=False
        ).reset_index()
        Apdex_df.columns = ['point', 'apdex']
        return Apdex_df
    
    def calculate_error(self):
        Error_df = self.df[
            (self.df['language'] == 'java') &
            (self.df['app_name'] == '[GMonit] Collector') &
            (self.df['scope'].isna()) &
            (self.df['name'].isin(['HttpDispatcher', 'Errors/allWeb']))
        ]
        Error_df = Error_df.groupby(self.time_col, group_keys=False).apply(
            lambda x: x[x['name'] == 'Errors/allWeb']['call_count'].sum() / 
                      x[x['name'] == 'HttpDispatcher']['call_count'].sum(), include_groups=False
        ).reset_index()
        Error_df.columns = ['point', 'error_rate']
        return Error_df

    def show_needed_interval(self, df, start_time, end_time):
        df_to_show = df[
            (df[self.time_col] >= start_time) & 
            (df[self.time_col] <= end_time)
        ]
        return df_to_show

class AnomalyDetector:
    def __init__(self, contamination=0.008):
        self.model = IsolationForest(contamination=contamination, random_state=42)
        
    def fit_predict(self, data):
        self.model.fit(data)
        scores = self.model.decision_function(data)
        predictions = self.model.predict(data)
        return predictions, scores

class App:
    def __init__(self):
        self.file_processor = None

    def run(self):
        st.title("RedLab Hack")
        st.header("Team: ikanam_chipi_chipi")

        file_path = st.text_input("Enter the path to your metrics_collector.tsv file")
        ready_path_button = st.button("Load data")

        if file_path and ready_path_button:
            try:
                self.file_processor = FileProcessor(file_path)
                self.file_processor.load_data()
                st.session_state.file_processor = self.file_processor
                st.session_state.df = self.file_processor.df
                st.session_state.time_col = self.file_processor.time_col

                # Calculating all proposed metrics
                throughput_df = self.file_processor.calculate_throughput()
                web_response_df = self.file_processor.calculate_web_response()
                apdex_df = self.file_processor.calculate_apdex()
                error_df = self.file_processor.calculate_error()

                # Merging all metrics
                combined_df = throughput_df.merge(web_response_df, on='point')
                combined_df = combined_df.merge(apdex_df, on='point')
                combined_df = combined_df.merge(error_df, on='point')

                st.session_state.combined_df = combined_df

                # Seasonal preprocessing for all metrics
                for col in ['sum_call_count', 'web_response', 'apdex', 'error_rate']:
                    decomposition = seasonal_decompose(combined_df[col], period=1440, model='additive')
                    combined_df[f'seasonally_adjusted_{col}'] = combined_df[col] - decomposition.seasonal

                # Detecting anomalies in the data
                anomaly_detector = AnomalyDetector()
                combined_data = combined_df[[f'seasonally_adjusted_{col}' for col in ['sum_call_count', 'web_response', 'apdex', 'error_rate']]].copy()
                st.session_state.combined_data = combined_data
                predictions, scores = anomaly_detector.fit_predict(combined_data)
                combined_df['anomaly'] = predictions
                combined_df['anomaly'] = combined_df['anomaly'].apply(lambda x: x == -1)  # Change -1 to True, 1 to False
                combined_df['anomaly_score'] = (1 - (scores - scores.min()) / (scores.max() - scores.min())).round(2)

                # SHAP explanation
                explainer = shap.TreeExplainer(anomaly_detector.model)
                shap_values = explainer.shap_values(combined_data)

                st.session_state.shap_values = shap_values
                st.session_state.expected_values = explainer.expected_value

            except Exception as e:
                st.error(f"An error occurred: {e}")

        if 'combined_df' in st.session_state:
            min_time = st.session_state.combined_df[st.session_state.time_col].min()
            max_time = st.session_state.combined_df[st.session_state.time_col].max()

            start_time, end_time = st.slider(
                "Select time interval",
                min_value=min_time.to_pydatetime(),
                max_value=max_time.to_pydatetime(),
                value=(min_time.replace(hour=0, minute=0).to_pydatetime(), max_time.to_pydatetime())
            )

            combined_interval_df = st.session_state.file_processor.show_needed_interval(
                st.session_state.combined_df, start_time, end_time
            )

            st.info("Combined Metrics Data:")
            st.dataframe(combined_interval_df[[st.session_state.time_col]+[f'seasonally_adjusted_{col}' for col in ['sum_call_count', 'web_response', 'apdex', 'error_rate']]+['anomaly', 'anomaly_score']])

            def plot_with_anomalies(df, y_col, title):
                fig = go.Figure()

                # Line plot for the metric
                fig.add_trace(go.Scatter(
                    x=df['point'], y=df[y_col],
                    mode='lines', name=y_col,
                    line=dict(color='gray')
                ))

                # Red dots for anomalies
                anomalies = df[df['anomaly']]
                fig.add_trace(go.Scatter(
                    x=anomalies['point'], y=anomalies[y_col],
                    mode='markers', name='Anomalies',
                    marker=dict(color='red', size=5)
                ))

                fig.update_layout(title=title, xaxis_title='Time', yaxis_title=y_col.capitalize(), showlegend=False)
                return fig

            show_throughput_fig = st.checkbox("Show Throughput and seasonal component to understand how preprocessing works")

            if show_throughput_fig:
                for col in ['sum_call_count', 'seasonality']:
                    fig = plot_with_anomalies(
                        combined_interval_df, 
                        col, 
                        f'{col.capitalize()}'
                    )
                    st.plotly_chart(fig)

            throughput_fig_adj = plot_with_anomalies(
                combined_interval_df, 
                'seasonally_adjusted_sum_call_count', 
                'Throughput Anomalies'
            )

            st.plotly_chart(throughput_fig_adj)

            web_response_fig = plot_with_anomalies(
                combined_interval_df, 
                'seasonally_adjusted_web_response', 
                'Web Response Anomalies'
            )
            st.plotly_chart(web_response_fig)

            apdex_fig = plot_with_anomalies(
                combined_interval_df, 
                'seasonally_adjusted_apdex', 
                'APDEX Anomalies'
            )
            st.plotly_chart(apdex_fig)

            error_fig = plot_with_anomalies(
                combined_interval_df, 
                'seasonally_adjusted_error_rate', 
                'Error Rate Anomalies'
            )
            st.plotly_chart(error_fig)

            show_shap = st.checkbox("Explanation of model decision")
            if show_shap:
                # SHAP Force Plot
                if 'combined_data' in st.session_state and 'shap_values' in st.session_state and 'expected_values' in st.session_state:
                    date = st.selectbox("Choose exact timestamp to get explanation of model's decision:", combined_interval_df.point.dt.strftime('%Y-%m-%d %H:%M:%S'))
                    selected_date = pd.to_datetime(date)
                    if selected_date in combined_interval_df.point.values:
                        index = combined_interval_df[combined_interval_df['point'] == selected_date].index[0]
                        shap.initjs()
                        shap_force_plot = shap.force_plot(
                            st.session_state.expected_values, 
                            st.session_state.shap_values[index], 
                            st.session_state.combined_data.iloc[index]
                        )

                        # Save the force plot as an HTML file and display it
                        shap_html = f"<head>{shap.getjs()}</head><body>{shap_force_plot.html()}</body>"
                        components.html(shap_html)
                    else:
                        st.error("Selected timestamp is not in the data range.")

if __name__ == "__main__":
    app = App()
    app.run()
