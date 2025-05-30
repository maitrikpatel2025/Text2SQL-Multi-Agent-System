import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Import Plotly if you intend to use it, but current plotting functions use Matplotlib
# import plotly.express as px
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Literal
import re
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class GraphPlottingAgent:
    """
    Agent that analyzes user question, explanation, table data, and graph suggestion
    to generate appropriate visualizations using matplotlib/seaborn.
    """

    def __init__(self, openai_api_key: str = None):
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            api_key=self.openai_api_key
        )

        # Set default style for Matplotlib plots
        plt.style.use('seaborn-v0_8-darkgrid') # A generally nice style
        # sns.set_palette("husl") # sns.set_palette is usually for seaborn-specific plots

    def analyze_and_plot(self, user_question: str, explanation: str, table_data: Any, graph_suggestion: str) -> Dict[str, Any]:
        """
        Main method that analyzes the input and generates appropriate visualizations
        """
        try:
            # Step 1: Convert table data to DataFrame if possible
            df = self._prepare_dataframe(table_data)

            if df is None or df.empty:
                return {
                    "success": False,
                    "error": "No valid data available for plotting",
                    "plot_analysis": {"plot_type": "none"},
                    "plot_result": {"plot_created": False, "message": "No data"}
                }

            # Step 2: Analyze what type of plot to create
            plot_analysis = self._analyze_plot_requirements(
                user_question, explanation, df, graph_suggestion
            )

            # Step 3: Generate the plot based on analysis
            # Ensure plot_analysis has a plot_type, default to "none" if not present
            if "plot_type" not in plot_analysis:
                plot_analysis["plot_type"] = "none"
                
            plot_result = self._generate_plot(df, plot_analysis)

            return {
                "success": True,
                "plot_analysis": plot_analysis,
                "plot_result": plot_result, # This will now contain the figure object if successful
                "dataframe_info": {
                    "shape": df.shape,
                    "columns": list(df.columns),
                    "sample_data": df.head(3).to_dict('records') if len(df) > 0 else []
                }
            }

        except Exception as e:
            print(f"Error in analyze_and_plot: {e}") # Print error for server logs
            return {
                "success": False,
                "error": str(e),
                "plot_analysis": {"plot_type": "error"},
                "plot_result": {"plot_created": False, "error": str(e)}
            }

    def _prepare_dataframe(self, table_data: Any) -> Optional[pd.DataFrame]:
        """Convert various table data formats to pandas DataFrame"""
        try:
            if table_data == "NA" or table_data is None:
                return None

            if isinstance(table_data, list) and len(table_data) > 0:
                if all(isinstance(item, dict) for item in table_data): # Check all items are dicts
                    df = pd.DataFrame(table_data)
                else:
                    print("Warning: table_data is a list but not all items are dictionaries.")
                    return None
            elif isinstance(table_data, dict):
                df = pd.DataFrame([table_data])
            elif isinstance(table_data, pd.DataFrame): # Already a DataFrame
                df = table_data
            else:
                print(f"Warning: table_data is of unhandled type: {type(table_data)}")
                return None

            if df.empty:
                return None

            # Clean column names
            df.columns = df.columns.astype(str).str.strip() # Ensure column names are strings

            # Try to convert date columns
            date_like_columns = ['time', 'date', 'timestamp'] # Add common variations
            for col in df.columns:
                if col.lower() in date_like_columns or 'date' in col.lower() or 'time' in col.lower():
                    try:
                        # Attempt conversion, but don't fail if it's not uniformly a date
                        converted_col = pd.to_datetime(df[col], errors='coerce')
                        # Only assign back if a significant portion could be converted (e.g., >50%)
                        # or if the original type wasn't already datetime
                        if not pd.api.types.is_datetime64_any_dtype(df[col]):
                            if converted_col.notna().sum() > 0: # If at least one value converted
                                df[col] = converted_col
                    except Exception: # Broad exception as date parsing is tricky
                        pass # Keep original if conversion fails

            # Convert numeric columns (only if object type)
            for col in df.columns:
                if df[col].dtype == 'object':
                    try:
                        numeric_series = pd.to_numeric(df[col], errors='coerce')
                        # Only assign if the column is not all NaNs after conversion
                        if not numeric_series.isna().all():
                            df[col] = numeric_series
                    except Exception:
                        pass # Keep original if conversion fails
            return df

        except Exception as e:
            print(f"Error preparing DataFrame: {e}")
            return None

    def _analyze_plot_requirements(self, user_question: str, explanation: str, df: pd.DataFrame, graph_suggestion: str) -> Dict[str, Any]:
        """Use LLM to analyze what type of plot should be created"""

        analysis_prompt_template = ChatPromptTemplate.from_messages([
            ("system", """
            You are a data visualization expert. Analyze the user question, explanation, and data structure to determine the best visualization approach for a Matplotlib-based plot.
            Return a JSON object with these fields:
            - plot_type: "bar", "line", "scatter", "pie", "histogram", "box", "area", "none" (Matplotlib compatible)
            - x_column: column name for x-axis (or null). Must be one of {columns}.
            - y_column: column name for y-axis (or null). Must be one of {columns}.
            - y_columns: LIST of column names for y-axis (for multi-line, grouped bar, etc.). Each must be one of {columns}.
            - group_by: column for grouping/coloring (or null). Must be one of {columns}.
            - aggregation_needed: boolean - if data needs aggregation.
            - aggregation_method: "sum", "count", "mean", "max", "min" (if needed).
            - title: suggested plot title.
            - x_label: x-axis label.
            - y_label: y-axis label.
            - plot_style: Always "matplotlib".
            - additional_options: (e.g., {{ "stacked": true }} for bar charts, {{ "log_scale_y": true }} )

            COLUMN SELECTION RULES:
            - Ensure x_column, y_column, y_columns, and group_by are actual column names from the provided list: {columns}.
            - If a suggested column is not in the list, pick the closest valid one or set to null.

            PLOT TYPE SELECTION (Matplotlib focus):
            - "bar": Comparisons, counts. For grouped bars, use x_column for categories, y_columns for series.
            - "line": Trends over time. y_columns can be used for multiple lines on the same plot.
            - "scatter": Relationships between two numeric variables (x_column, y_column).
            - "pie": Parts of a whole for a single categorical variable (x_column for labels, y_column for values).
            - "histogram": Frequency distribution of a single numeric variable (y_column).
            - "box": Statistical summary of a numeric variable (y_column), possibly grouped by a category (x_column).
            - "area": Cumulative trends, similar to line but filled.
            - "none": If data is not suitable for visualization or columns are missing.
 
            AGGREGATION:
            If `aggregation_needed` is true, specify `group_by` (the categorical column for grouping) and `y_column` (the numeric column to aggregate) and `aggregation_method`.
            The `x_column` will then typically be the `group_by` column after aggregation.
            """),
            ("human", """
            User Question: {user_question}
            Explanation: {explanation}
            Graph Suggestion: {graph_suggestion}

            DataFrame Details:
            Columns: {columns}
            Data types: {dtypes}
            Sample rows (first 3): {sample_data}

            Please analyze and suggest the best Matplotlib visualization. Ensure all column names chosen exist in the 'Columns' list.
            """)
        ])

        try:
            chain = analysis_prompt_template | self.llm | JsonOutputParser()
            df_cols = list(df.columns)
            result = chain.invoke({
                "user_question": user_question,
                "explanation": explanation,
                "graph_suggestion": graph_suggestion,
                "columns": df_cols,
                "dtypes": df.dtypes.astype(str).to_dict(), # Convert dtypes to string for JSON
                "sample_data": df.head(3).to_dict('records')
            })
            # Basic validation of returned column names
            for key in ["x_column", "y_column", "group_by"]:
                if result.get(key) and result[key] not in df_cols:
                    print(f"Warning: LLM suggested invalid column '{result[key]}' for '{key}'. Setting to None.")
                    result[key] = None
            if result.get("y_columns"):
                result["y_columns"] = [col for col in result["y_columns"] if col in df_cols]

            return result

        except Exception as e:
            print(f"Error in LLM-based plot analysis: {e}. Falling back to basic analysis.")
            return self._fallback_plot_analysis(df, user_question)

    def _fallback_plot_analysis(self, df: pd.DataFrame, user_question: str) -> Dict[str, Any]:
        """Fallback plot analysis when LLM fails, focusing on Matplotlib."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        date_cols = df.select_dtypes(include=['datetime64[ns]']).columns.tolist()

        analysis = {
            "plot_type": "none", "x_column": None, "y_column": None, "y_columns": [],
            "group_by": None, "aggregation_needed": False, "aggregation_method": None,
            "title": "Data Plot", "x_label": None, "y_label": None, "plot_style": "matplotlib",
            "additional_options": {}
        }

        if len(date_cols) > 0 and len(numeric_cols) > 0:
            analysis.update({
                "plot_type": "line", "x_column": date_cols[0], "y_column": numeric_cols[0],
                "title": f"{numeric_cols[0]} over {date_cols[0]}",
                "x_label": date_cols[0], "y_label": numeric_cols[0]
            })
        elif len(categorical_cols) > 0 and len(numeric_cols) > 0:
            analysis.update({
                "plot_type": "bar", "x_column": categorical_cols[0], "y_column": numeric_cols[0],
                "title": f"{numeric_cols[0]} by {categorical_cols[0]}",
                "x_label": categorical_cols[0], "y_label": numeric_cols[0]
            })
        elif len(numeric_cols) >= 2:
            analysis.update({
                "plot_type": "scatter", "x_column": numeric_cols[0], "y_column": numeric_cols[1],
                "title": f"{numeric_cols[1]} vs {numeric_cols[0]}",
                "x_label": numeric_cols[0], "y_label": numeric_cols[1]
            })
        elif len(numeric_cols) == 1:
             analysis.update({
                "plot_type": "histogram", "y_column": numeric_cols[0],
                "title": f"Distribution of {numeric_cols[0]}",
                "x_label": numeric_cols[0], "y_label": "Frequency"
            })
        return analysis

    def _generate_plot(self, df: pd.DataFrame, plot_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate the actual plot based on analysis"""
        plot_type = plot_analysis.get("plot_type", "none")
        fig = None # Initialize fig to handle cases where plot creation fails

        if plot_type == "none":
            return {"plot_created": False, "message": "No suitable plot type determined", "figure": None}

        try:
            # Prepare data if aggregation is needed
            if plot_analysis.get("aggregation_needed", False) and plot_analysis.get("group_by"):
                agg_df = self._aggregate_data(df, plot_analysis)
                if agg_df is None or agg_df.empty: # Aggregation might fail or result in empty
                     return {"plot_created": False, "message": "Aggregation resulted in no data", "figure": None}
                df_to_plot = agg_df
                # Update x_column to be the group_by column after aggregation, if not already set
                if not plot_analysis.get("x_column") and plot_analysis.get("group_by"):
                    plot_analysis["x_column"] = plot_analysis["group_by"]
            else:
                df_to_plot = df.copy()


            # Generate plot based on type
            if plot_type == "bar":
                return self._create_bar_plot(df_to_plot, plot_analysis)
            elif plot_type == "line":
                return self._create_line_plot(df_to_plot, plot_analysis)
            elif plot_type == "scatter":
                return self._create_scatter_plot(df_to_plot, plot_analysis)
            elif plot_type == "pie":
                return self._create_pie_plot(df_to_plot, plot_analysis)
            elif plot_type == "histogram":
                return self._create_histogram(df_to_plot, plot_analysis)
            elif plot_type == "box":
                return self._create_box_plot(df_to_plot, plot_analysis)
            # Heatmap typically uses the correlation of the original numeric df
            elif plot_type == "heatmap":
                return self._create_heatmap(df, plot_analysis) # Pass original df for heatmap
            elif plot_type == "area":
                return self._create_area_plot(df_to_plot, plot_analysis)
            else:
                return {"plot_created": False, "message": f"Plot type '{plot_type}' not implemented", "figure": None}

        except Exception as e:
            print(f"Error during plot generation for type '{plot_type}': {e}")
            if fig: # If a figure object was created before the error
                plt.close(fig)
            return {"plot_created": False, "error": str(e), "figure": None}

    def _aggregate_data(self, df: pd.DataFrame, plot_analysis: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Aggregate data as needed"""
        group_by_col = plot_analysis.get("group_by")
        agg_method = plot_analysis.get("aggregation_method", "sum")
        # y_column is the column to be aggregated.
        # If multiple y_columns, this simple aggregation might need adjustment or handle one primary y_column.
        val_col = plot_analysis.get("y_column") # The numeric column to aggregate

        if not group_by_col or not val_col:
            print("Aggregation skipped: group_by column or value column for aggregation not specified.")
            return df # Return original df if params are missing

        if group_by_col not in df.columns or val_col not in df.columns:
            print(f"Aggregation skipped: '{group_by_col}' or '{val_col}' not in DataFrame columns.")
            return df

        try:
            if agg_method == "count":
                # For count, we count occurrences of group_by_col, or count distinct items in val_col grouped by group_by_col
                # Let's assume it's counting rows per group_by_col, and val_col becomes the count name
                agg_df = df.groupby(group_by_col).size().reset_index(name=val_col)
            else:
                agg_func_map = {"sum": "sum", "mean": "mean", "average": "mean", "max": "max", "min": "min"}
                selected_agg_func = agg_func_map.get(agg_method.lower(), "sum")

                # Ensure val_col is numeric before aggregation
                if not pd.api.types.is_numeric_dtype(df[val_col]):
                    print(f"Warning: Column '{val_col}' is not numeric. Attempting to convert for aggregation.")
                    df[val_col] = pd.to_numeric(df[val_col], errors='coerce')
                    if df[val_col].isnull().all():
                        print(f"Error: Column '{val_col}' could not be converted to numeric for aggregation.")
                        return None
                
                agg_df = df.groupby(group_by_col, as_index=False)[val_col].agg(selected_agg_func)
            return agg_df
        except Exception as e:
            print(f"Error during data aggregation: {e}")
            return None


    def _create_bar_plot(self, df: pd.DataFrame, plot_analysis: Dict[str, Any]) -> Dict[str, Any]:
        x_col = plot_analysis.get("x_column")
        y_col = plot_analysis.get("y_column") # Could also be a list for grouped/stacked
        title = plot_analysis.get("title", "Bar Chart")
        fig = None
        try:
            if not x_col or not y_col or x_col not in df.columns:
                return {"plot_created": False, "message": "Missing X or Y column for bar plot", "figure": None}
            # Check if y_col is a list (for grouped/stacked) or a single column string
            if isinstance(y_col, list):
                if not all(col in df.columns for col in y_col):
                    return {"plot_created": False, "message": "One or more Y columns for bar plot not found in data", "figure": None}
            elif y_col not in df.columns:
                 return {"plot_created": False, "message": f"Y column '{y_col}' for bar plot not found in data", "figure": None}


            fig, ax = plt.subplots(figsize=(10, 6))
            df_sorted = df.sort_values(by=y_col[0] if isinstance(y_col, list) else y_col, ascending=False)

            sns.barplot(x=x_col, y=(y_col[0] if isinstance(y_col, list) else y_col), data=df_sorted, ax=ax, palette="viridis") # Using a seaborn palette
            # If y_col is a list for grouped/stacked, df.plot(kind='bar') is often easier
            # For simplicity, this example handles single y_col with sns.barplot.
            # df_sorted.plot(x=x_col, y=y_col, kind='bar', ax=ax, 
            #                stacked=plot_analysis.get("additional_options", {}).get("stacked", False))


            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel(plot_analysis.get("x_label", x_col))
            ax.set_ylabel(plot_analysis.get("y_label", y_col if isinstance(y_col, str) else "Values"))
            plt.xticks(rotation=45, ha='right')

            # Add value labels (works best for single y_col)
            if not isinstance(y_col, list):
                for p in ax.patches:
                    ax.annotate(f'{p.get_height():.1f}',
                                (p.get_x() + p.get_width() / 2., p.get_height()),
                                ha='center', va='center',
                                xytext=(0, 9),
                                textcoords='offset points')
            plt.tight_layout()
            # REMOVE plt.show()
            return {"plot_created": True, "plot_type": "bar", "title": title, "data_points": len(df_sorted), "figure": fig}
        except Exception as e:
            if fig: plt.close(fig)
            return {"plot_created": False, "error": str(e), "figure": None}

    def _create_line_plot(self, df: pd.DataFrame, plot_analysis: Dict[str, Any]) -> Dict[str, Any]:
        x_col = plot_analysis.get("x_column")
        y_cols = plot_analysis.get("y_columns") if plot_analysis.get("y_columns") else [plot_analysis.get("y_column")]
        group_by = plot_analysis.get("group_by") # For coloring lines by category
        title = plot_analysis.get("title", "Line Chart")
        fig = None
        try:
            if not x_col or not y_cols or x_col not in df.columns or not all(yc in df.columns for yc in y_cols if yc):
                return {"plot_created": False, "message": "Missing X or Y column(s) for line plot", "figure": None}
            y_cols = [yc for yc in y_cols if yc] # Filter out None values if any
            if not y_cols: return {"plot_created": False, "message": "No valid Y columns for line plot", "figure": None}


            fig, ax = plt.subplots(figsize=(12, 6))
            df_sorted = df.sort_values(by=x_col)
 
            if group_by and group_by in df.columns:
                for group_val in df_sorted[group_by].unique():
                    group_data = df_sorted[df_sorted[group_by] == group_val]
                    for y_col_single in y_cols:
                        ax.plot(group_data[x_col], group_data[y_col_single], marker='o', linewidth=2, label=f"{group_val} - {y_col_single}")
                ax.legend(title=group_by)
            else:
                for y_col_single in y_cols:
                    ax.plot(df_sorted[x_col], df_sorted[y_col_single], marker='o', linewidth=2, label=y_col_single)
                if len(y_cols) > 1:
                    ax.legend()

            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel(plot_analysis.get("x_label", x_col))
            ax.set_ylabel(plot_analysis.get("y_label", ", ".join(y_cols)))
            plt.grid(True, alpha=0.5)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            # REMOVE plt.show()
            return {"plot_created": True, "plot_type": "line", "title": title, "data_points": len(df_sorted), "figure": fig}
        except Exception as e:
            if fig: plt.close(fig)
            return {"plot_created": False, "error": str(e), "figure": None}

    def _create_scatter_plot(self, df: pd.DataFrame, plot_analysis: Dict[str, Any]) -> Dict[str, Any]:
        x_col = plot_analysis.get("x_column")
        y_col = plot_analysis.get("y_column")
        group_by = plot_analysis.get("group_by") # For color encoding
        title = plot_analysis.get("title", "Scatter Plot")
        fig = None
        try:
            if not x_col or not y_col or x_col not in df.columns or y_col not in df.columns:
                return {"plot_created": False, "message": "Missing X or Y column for scatter plot", "figure": None}

            fig, ax = plt.subplots(figsize=(10, 8))
            sns.scatterplot(x=x_col, y=y_col, hue=group_by if group_by in df.columns else None, data=df, ax=ax, s=60, alpha=0.7, palette="viridis")

            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel(plot_analysis.get("x_label", x_col))
            ax.set_ylabel(plot_analysis.get("y_label", y_col))
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            # REMOVE plt.show()
            return {"plot_created": True, "plot_type": "scatter", "title": title, "data_points": len(df), "figure": fig}
        except Exception as e:
            if fig: plt.close(fig)
            return {"plot_created": False, "error": str(e), "figure": None}

    def _create_pie_plot(self, df: pd.DataFrame, plot_analysis: Dict[str, Any]) -> Dict[str, Any]:
        labels_col = plot_analysis.get("x_column") # Column for labels
        values_col = plot_analysis.get("y_column") # Column for values
        title = plot_analysis.get("title", "Pie Chart")
        fig = None
        try:
            if not labels_col or labels_col not in df.columns:
                return {"plot_created": False, "message": "Missing labels column for pie chart", "figure": None}

            if values_col and values_col in df.columns:
                # Ensure values_col is numeric
                if not pd.api.types.is_numeric_dtype(df[values_col]):
                     return {"plot_created": False, "message": f"Values column '{values_col}' must be numeric for pie chart", "figure": None}
                # Check for negative values if summing, as pie charts don't support them
                if df[values_col].min() < 0:
                    return {"plot_created": False, "message": f"Values column '{values_col}' contains negative values, unsuitable for pie chart.", "figure": None}
                pie_data = df.set_index(labels_col)[values_col]
            else: # If no specific values column, count occurrences of labels_col
                pie_data = df[labels_col].value_counts()

            if pie_data.empty or pie_data.sum() == 0:
                 return {"plot_created": False, "message": "No data or zero sum for pie chart", "figure": None}


            fig, ax = plt.subplots(figsize=(10, 8))
            wedges, texts, autotexts = ax.pie(pie_data, labels=pie_data.index,
                                            autopct='%1.1f%%', startangle=140,
                                            pctdistance=0.85,
                                            colors=sns.color_palette("viridis", len(pie_data)))
            # Improve aesthetics
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
            ax.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
            ax.set_title(title, fontsize=14, fontweight='bold')
            plt.tight_layout()
            # REMOVE plt.show()
            return {"plot_created": True, "plot_type": "pie", "title": title, "categories": len(pie_data), "figure": fig}
        except Exception as e:
            if fig: plt.close(fig)
            return {"plot_created": False, "error": str(e), "figure": None}

    def _create_histogram(self, df: pd.DataFrame, plot_analysis: Dict[str, Any]) -> Dict[str, Any]:
        data_col = plot_analysis.get("y_column") # Column for which to plot histogram
        title = plot_analysis.get("title", "Histogram")
        fig = None
        try:
            if not data_col or data_col not in df.columns:
                numeric_cols = df.select_dtypes(include=np.number).columns
                if not numeric_cols.empty: data_col = numeric_cols[0]
                else: return {"plot_created": False, "message": "No numeric column specified or found for histogram", "figure": None}
            
            if not pd.api.types.is_numeric_dtype(df[data_col]):
                 return {"plot_created": False, "message": f"Column '{data_col}' must be numeric for histogram", "figure": None}


            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(df[data_col].dropna(), kde=True, ax=ax, bins=plot_analysis.get("additional_options",{}).get("bins", 20), color="skyblue")
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel(plot_analysis.get("x_label", data_col))
            ax.set_ylabel(plot_analysis.get("y_label", 'Frequency'))
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            # REMOVE plt.show()
            return {"plot_created": True, "plot_type": "histogram", "title": title, "data_points": len(df[data_col].dropna()), "figure": fig}
        except Exception as e:
            if fig: plt.close(fig)
            return {"plot_created": False, "error": str(e), "figure": None}

    def _create_box_plot(self, df: pd.DataFrame, plot_analysis: Dict[str, Any]) -> Dict[str, Any]:
        y_col = plot_analysis.get("y_column") # Numeric data column
        x_col = plot_analysis.get("x_column") # Categorical column for grouping (optional)
        title = plot_analysis.get("title", "Box Plot")
        fig = None
        try:
            if not y_col or y_col not in df.columns:
                numeric_cols = df.select_dtypes(include=np.number).columns
                if not numeric_cols.empty: y_col = numeric_cols[0]
                else: return {"plot_created": False, "message": "No numeric Y column for box plot", "figure": None}

            if not pd.api.types.is_numeric_dtype(df[y_col]):
                 return {"plot_created": False, "message": f"Y Column '{y_col}' must be numeric for box plot", "figure": None}
            if x_col and x_col not in df.columns:
                print(f"Warning: Grouping column '{x_col}' not found. Creating non-grouped box plot.")
                x_col = None # Proceed without grouping

            fig, ax = plt.subplots(figsize=(10, 6))
            if x_col:
                sns.boxplot(x=x_col, y=y_col, data=df, ax=ax, palette="pastel")
                ax.set_xlabel(plot_analysis.get("x_label", x_col))
            else:
                sns.boxplot(y=y_col, data=df, ax=ax, palette="pastel")
                ax.set_xlabel(None) # No x-label if not grouped

            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_ylabel(plot_analysis.get("y_label", y_col))
            plt.tight_layout()
            # REMOVE plt.show()
            return {"plot_created": True, "plot_type": "box", "title": title, "figure": fig}
        except Exception as e:
            if fig: plt.close(fig)
            return {"plot_created": False, "error": str(e), "figure": None}

    def _create_heatmap(self, df: pd.DataFrame, plot_analysis: Dict[str, Any]) -> Dict[str, Any]:
        title = plot_analysis.get("title", "Correlation Heatmap")
        fig = None
        try:
            numeric_df = df.select_dtypes(include=[np.number])
            if numeric_df.shape[1] < 2:
                return {"plot_created": False, "message": "Need at least 2 numeric columns for correlation heatmap", "figure": None}

            correlation_matrix = numeric_df.corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                        square=True, fmt='.2f', linewidths=.5, ax=ax)
            ax.set_title(title, fontsize=14, fontweight='bold')
            plt.tight_layout()
            # REMOVE plt.show()
            return {"plot_created": True, "plot_type": "heatmap", "title": title, "variables": list(numeric_df.columns), "figure": fig}
        except Exception as e:
            if fig: plt.close(fig)
            return {"plot_created": False, "error": str(e), "figure": None}

    def _create_area_plot(self, df: pd.DataFrame, plot_analysis: Dict[str, Any]) -> Dict[str, Any]:
        x_col = plot_analysis.get("x_column")
        y_col = plot_analysis.get("y_column") # For simple area plot, one y-column
        # For stacked area, y_columns would be a list and df might need to be wide or reshaped
        title = plot_analysis.get("title", "Area Chart")
        fig = None
        try:
            if not x_col or not y_col or x_col not in df.columns or y_col not in df.columns:
                return {"plot_created": False, "message": "Missing X or Y column for area plot", "figure": None}
            
            if not pd.api.types.is_numeric_dtype(df[y_col]):
                 return {"plot_created": False, "message": f"Y Column '{y_col}' must be numeric for area plot", "figure": None}


            fig, ax = plt.subplots(figsize=(12, 6))
            df_sorted = df.sort_values(by=x_col)

            # Simple area plot for one y-column
            ax.fill_between(df_sorted[x_col], df_sorted[y_col], alpha=0.6, color="skyblue")
            ax.plot(df_sorted[x_col], df_sorted[y_col], linewidth=2, color="steelblue") # Add line for better definition

            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel(plot_analysis.get("x_label", x_col))
            ax.set_ylabel(plot_analysis.get("y_label", y_col))
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            # REMOVE plt.show()
            return {"plot_created": True, "plot_type": "area", "title": title, "data_points": len(df_sorted), "figure": fig}
        except Exception as e:
            if fig: plt.close(fig)
            return {"plot_created": False, "error": str(e), "figure": None}