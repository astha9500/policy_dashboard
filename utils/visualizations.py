"""
Visualization Utilities for Two-Child Policy Survey Analysis
Contains all chart creation functions using Plotly
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Optional, List, Dict


# Color schemes
COLOR_SCHEMES = {
    'support': ['#d62728', '#ff7f0e', '#ffd700', '#2ca02c', '#1f77b4'],  # Red to Blue
    'concern': ['#1f77b4', '#2ca02c', '#ffd700', '#ff7f0e', '#d62728'],  # Blue to Red
    'diverging': ['#d62728', '#ff9999', '#e0e0e0', '#99ccff', '#1f77b4'],  # Red-Gray-Blue
    'sequential': px.colors.sequential.Blues,
    'categorical': px.colors.qualitative.Set2
}


class SurveyVisualizer:
    """Main class for creating survey visualizations"""
    
    def __init__(self, theme: str = 'plotly_white'):
        """
        Initialize visualizer
        
        Args:
            theme: Plotly theme to use
        """
        self.theme = theme
        self.default_height = 500
        self.default_width = None
    
    def create_demographic_bar_chart(self, 
                                     df: pd.DataFrame, 
                                     x_col: str, 
                                     y_col: str,
                                     title: str,
                                     x_label: Optional[str] = None,
                                     y_label: Optional[str] = None,
                                     color_scale: str = 'Blues',
                                     add_neutral_line: bool = True) -> go.Figure:
        """
        Create a bar chart for demographic analysis
        
        Args:
            df: Dataframe with data
            x_col: Column for x-axis
            y_col: Column for y-axis
            title: Chart title
            x_label: Custom x-axis label
            y_label: Custom y-axis label
            color_scale: Color scale to use
            add_neutral_line: Add horizontal line at neutral (3.0)
            
        Returns:
            Plotly figure
        """
        # Drop NaNs to avoid empty categories
        clean_df = df[[x_col, y_col]].dropna()
        agg_df = clean_df.groupby(x_col)[y_col].mean().reset_index()
        
        fig = px.bar(
            agg_df, 
            x=x_col, 
            y=y_col,
            title=title,
            labels={
                x_col: x_label or x_col,
                y_col: y_label or y_col
            },
            color=y_col,
            color_continuous_scale=color_scale,
            template=self.theme
        )
        
        if add_neutral_line and df[y_col].min() <= 3 <= df[y_col].max():
            fig.add_hline(
                y=3, 
                line_dash="dash", 
                line_color="red",
                annotation_text="Neutral (3.0)",
                annotation_position="right"
            )
        
        fig.update_layout(
            height=self.default_height,
            showlegend=False,
            hovermode='x unified'
        )
        
        return fig
    
    def create_grouped_bar_chart(self,
                                 df: pd.DataFrame,
                                 x_col: str,
                                 y_col: str,
                                 group_col: Optional[str] = None,
                                 title: str = "",
                                 x_label: Optional[str] = None,
                                 y_label: Optional[str] = None) -> go.Figure:
        """
        Create grouped bar chart
        
        Args:
            df: Dataframe with data
            x_col: Column for x-axis
            y_col: Column for y-axis
            group_col: Column for grouping
            title: Chart title
            x_label: Custom x-axis label
            y_label: Custom y-axis label
            
        Returns:
            Plotly figure
        """
        if group_col:
            agg_df = df.groupby([x_col, group_col])[y_col].mean().reset_index()
            fig = px.bar(
                agg_df,
                x=x_col,
                y=y_col,
                color=group_col,
                barmode='group',
                title=title,
                labels={
                    x_col: x_label or x_col,
                    y_col: y_label or y_col,
                    group_col: group_col
                },
                template=self.theme
            )
        else:
            agg_df = df.groupby(x_col)[y_col].mean().reset_index()
            fig = px.bar(
                agg_df,
                x=x_col,
                y=y_col,
                title=title,
                labels={
                    x_col: x_label or x_col,
                    y_col: y_label or y_col
                },
                template=self.theme
            )
        
        fig.update_layout(height=self.default_height)
        
        return fig
    
    def create_line_chart(self,
                         df: pd.DataFrame,
                         x_col: str,
                         y_col: str,
                         title: str,
                         x_label: Optional[str] = None,
                         y_label: Optional[str] = None,
                         add_markers: bool = True,
                         add_trendline: bool = False) -> go.Figure:
        """
        Create line chart with optional trendline
        
        Args:
            df: Dataframe with data
            x_col: Column for x-axis
            y_col: Column for y-axis
            title: Chart title
            x_label: Custom x-axis label
            y_label: Custom y-axis label
            add_markers: Add markers to line
            add_trendline: Add trendline
            
        Returns:
            Plotly figure
        """
        agg_df = df.groupby(x_col)[y_col].mean().reset_index()
        
        if add_trendline:
            fig = px.scatter(
                agg_df,
                x=x_col,
                y=y_col,
                trendline="ols",
                title=title,
                labels={
                    x_col: x_label or x_col,
                    y_col: y_label or y_col
                },
                template=self.theme
            )
        else:
            fig = px.line(
                agg_df,
                x=x_col,
                y=y_col,
                title=title,
                labels={
                    x_col: x_label or x_col,
                    y_col: y_label or y_col
                },
                markers=add_markers,
                template=self.theme
            )
        
        fig.add_hline(y=3, line_dash="dash", line_color="red", annotation_text="Neutral")
        fig.update_layout(height=self.default_height)
        
        return fig
    
    def create_diverging_stacked_bar(self,
                                     df: pd.DataFrame,
                                     column: str,
                                     title: str,
                                     show_percentages: bool = True) -> go.Figure:
        """
        Create diverging stacked bar chart for Likert scale data
        
        Args:
            df: Dataframe with data
            column: Column containing Likert responses
            title: Chart title
            show_percentages: Show percentage labels
            
        Returns:
            Plotly figure
        """
        # Count responses
        counts = df[column].value_counts()
        total = len(df)
        
        # Define order and colors
        likert_order = ['Strongly Disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly Agree']
        colors = COLOR_SCHEMES['diverging']
        
        # Calculate percentages
        data_for_plot = []
        
        for i, response in enumerate(likert_order):
            if response in counts.index:
                pct = (counts[response] / total) * 100
                data_for_plot.append({
                    'response': response,
                    'percentage': pct,
                    'count': counts[response],
                    'color': colors[i]
                })
        
        # Create figure
        fig = go.Figure()
        
        for item in data_for_plot:
            fig.add_trace(go.Bar(
                name=item['response'],
                y=['Distribution'],
                x=[item['percentage']],
                orientation='h',
                marker=dict(color=item['color']),
                text=f"{item['percentage']:.1f}%" if show_percentages else "",
                textposition='inside',
                hovertemplate=f"{item['response']}<br>Count: {item['count']}<br>Percentage: {item['percentage']:.1f}%<extra></extra>"
            ))
        
        fig.update_layout(
            title=title,
            barmode='stack',
            xaxis_title="Percentage (%)",
            yaxis_title="",
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            height=300,
            template=self.theme
        )
        
        return fig
    
    def create_net_score_bar(self,
                            df: pd.DataFrame,
                            x_col: str,
                            sentiment_col: str,
                            title: str,
                            x_label: Optional[str] = None) -> go.Figure:
        """
        Create bar chart showing net scores (colored by positive/negative)
        
        Args:
            df: Dataframe with data
            x_col: Column for x-axis (demographic)
            sentiment_col: Column with sentiment scores
            title: Chart title
            x_label: Custom x-axis label
            
        Returns:
            Plotly figure
        """
        agg_df = df.groupby(x_col)[sentiment_col].mean().reset_index()
        agg_df['Color'] = agg_df[sentiment_col].apply(lambda x: 'Positive' if x > 0 else 'Negative')
        
        fig = px.bar(
            agg_df,
            x=x_col,
            y=sentiment_col,
            title=title,
            labels={
                x_col: x_label or x_col,
                sentiment_col: 'Net Score'
            },
            color='Color',
            color_discrete_map={'Positive': '#2ca02c', 'Negative': '#d62728'},
            template=self.theme
        )
        
        fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=2)
        fig.update_layout(height=self.default_height)
        
        return fig
    
    def create_correlation_heatmap(self,
                                   correlation_matrix: pd.DataFrame,
                                   title: str = "Statement Correlation Matrix") -> go.Figure:
        """
        Create correlation heatmap
        
        Args:
            correlation_matrix: Correlation matrix dataframe
            title: Chart title
            
        Returns:
            Plotly figure
        """
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale='RdBu',
            zmid=0,
            text=np.round(correlation_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Statements",
            yaxis_title="Statements",
            height=600,
            template=self.theme
        )
        
        return fig
    
    def create_pie_chart(self,
                        df: pd.DataFrame,
                        column: str,
                        title: str) -> go.Figure:
        """
        Create pie chart for categorical distribution
        
        Args:
            df: Dataframe with data
            column: Column to visualize
            title: Chart title
            
        Returns:
            Plotly figure
        """
        counts = df[column].value_counts()
        
        fig = go.Figure(data=[go.Pie(
            labels=counts.index,
            values=counts.values,
            hole=0.3,
            textinfo='label+percent',
            hovertemplate='%{label}<br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
        )])
        
        fig.update_layout(
            title=title,
            height=self.default_height,
            template=self.theme
        )
        
        return fig
    
    def create_stacked_bar(self,
                          df: pd.DataFrame,
                          x_col: str,
                          y_col: str,
                          stack_col: str,
                          title: str) -> go.Figure:
        """
        Create stacked bar chart
        
        Args:
            df: Dataframe with data
            x_col: Column for x-axis
            y_col: Column for y-axis
            stack_col: Column for stacking
            title: Chart title
            
        Returns:
            Plotly figure
        """
        agg_df = df.groupby([x_col, stack_col])[y_col].mean().reset_index()
        
        fig = px.bar(
            agg_df,
            x=x_col,
            y=y_col,
            color=stack_col,
            title=title,
            barmode='stack',
            template=self.theme
        )
        
        fig.update_layout(height=self.default_height)
        
        return fig
    
    def create_box_plot(self,
                       df: pd.DataFrame,
                       x_col: str,
                       y_col: str,
                       title: str) -> go.Figure:
        """
        Create box plot for distribution analysis
        
        Args:
            df: Dataframe with data
            x_col: Column for x-axis (categories)
            y_col: Column for y-axis (values)
            title: Chart title
            
        Returns:
            Plotly figure
        """
        fig = px.box(
            df,
            x=x_col,
            y=y_col,
            title=title,
            template=self.theme
        )
        
        fig.update_layout(height=self.default_height)
        
        return fig


# ============================================================================
# MATPLOTLIB VISUALIZATION FUNCTIONS
# ============================================================================

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle


class MatplotlibVisualizer:
    """Matplotlib-based visualizer for survey data"""
    
    def __init__(self, style: str = 'seaborn-v0_8-darkgrid'):
        """
        Initialize matplotlib visualizer
        
        Args:
            style: Matplotlib style to use
        """
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
        
        self.figsize = (10, 6)
        sns.set_palette("husl")
    
    def create_demographic_bar_chart(self,
                                     df: pd.DataFrame,
                                     x_col: str,
                                     y_col: str,
                                     title: str,
                                     x_label: Optional[str] = None,
                                     y_label: Optional[str] = None,
                                     add_neutral_line: bool = False) -> Figure:
        """
        Create bar chart using matplotlib
        
        Args:
            df: Dataframe with data
            x_col: Column for x-axis
            y_col: Column for y-axis
            title: Chart title
            x_label: Custom x-axis label
            y_label: Custom y-axis label
            add_neutral_line: Add horizontal line at neutral (3.0)
            
        Returns:
            Matplotlib figure
        """
        agg_df = df.groupby(x_col)[y_col].mean().reset_index()
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Ensure categorical x-axis values are strings to avoid dtype issues
        x_labels = agg_df[x_col].astype(str)

        bars = ax.bar(x_labels, agg_df[y_col], 
                     color=sns.color_palette("Blues_d", len(agg_df)),
                     edgecolor='black', linewidth=1.2)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=10)
        
        if add_neutral_line:
            ax.axhline(y=3, color='red', linestyle='--', linewidth=2, 
                      label='Neutral (3.0)', alpha=0.7)
            ax.legend()
        
        ax.set_xlabel(x_label or x_col, fontsize=12, fontweight='bold')
        ax.set_ylabel(y_label or y_col, fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        return fig
    
    def create_line_chart(self,
                         df: pd.DataFrame,
                         x_col: str,
                         y_col: str,
                         title: str,
                         x_label: Optional[str] = None,
                         y_label: Optional[str] = None,
                         add_markers: bool = True,
                         add_trendline: bool = False) -> Figure:
        """
        Create line chart using matplotlib
        
        Args:
            df: Dataframe with data
            x_col: Column for x-axis
            y_col: Column for y-axis
            title: Chart title
            x_label: Custom x-axis label
            y_label: Custom y-axis label
            add_markers: Add markers to line
            add_trendline: Add polynomial trendline
            
        Returns:
            Matplotlib figure
        """
        agg_df = df.groupby(x_col)[y_col].mean().reset_index()
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        marker_style = 'o' if add_markers else None
        ax.plot(agg_df[x_col], agg_df[y_col], 
               marker=marker_style, linewidth=2.5, markersize=8,
               color='#1f77b4', label='Mean Score')
        
        if add_trendline and len(agg_df) > 2:
            # Add polynomial trendline
            x_numeric = range(len(agg_df))
            z = np.polyfit(x_numeric, agg_df[y_col], 2)
            p = np.poly1d(z)
            ax.plot(agg_df[x_col], p(x_numeric), 
                   linestyle='--', color='red', linewidth=2, 
                   label='Trendline', alpha=0.7)
        
        # Neutral reference line removed per user preference
        
        ax.set_xlabel(x_label or x_col, fontsize=12, fontweight='bold')
        ax.set_ylabel(y_label or y_col, fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        return fig

    def create_pos_neg_stacked_share(self,
                                     df: pd.DataFrame,
                                     group_col: str,
                                     likert_col: str,
                                     title: str,
                                     figsize: Optional[tuple] = None) -> Figure:
        """
        Stacked percentage bar by group with Negative/Neutral/Positive shares.
        Requires normalized Likert labels.
        """
        clean_df = df[[group_col, likert_col]].dropna()
        # Ensure strings for grouping labels
        clean_df[group_col] = clean_df[group_col].astype(str)

        groups = sorted(clean_df[group_col].unique().tolist(), key=lambda x: x)
        categories = [
            ("Negative", ["Strongly Disagree", "Disagree"], "#e74c3c"),  # Better red
            ("Neutral", ["Neutral"], "#f39c12"),  # Orange instead of grey
            ("Positive", ["Agree", "Strongly Agree"], "#27ae60"),  # Better green
        ]

        data = {name: [] for name, _, _ in categories}
        for g in groups:
            sub = clean_df[clean_df[group_col] == g]
            total = len(sub)
            for name, labels, _ in categories:
                pct = (sub[likert_col].isin(labels).sum() / total * 100) if total else 0
                data[name].append(pct)

        fig, ax = plt.subplots(figsize=(figsize or self.figsize))
        left = np.zeros(len(groups))
        for name, _, color in categories:
            ax.bar(groups, data[name], bottom=left, color=color, edgecolor='white', linewidth=1.0, label=name)
            left += np.array(data[name])

        ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
        ax.set_xlabel(group_col, fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_ylim(0, 100)
        ax.legend(loc='upper center', ncol=3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        return fig
    
    def create_diverging_stacked_bar(self,
                                     df: pd.DataFrame,
                                     column: str,
                                     title: str) -> Figure:
        """
        Create diverging stacked bar chart for Likert data using matplotlib
        
        Args:
            df: Dataframe with data
            column: Column containing Likert responses
            title: Chart title
            
        Returns:
            Matplotlib figure
        """
        # Count responses
        counts = df[column].value_counts()
        total = len(df)
        
        # Define order and colors
        likert_order = ['Strongly Disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly Agree']
        colors = ['#d62728', '#ff9999', '#e0e0e0', '#99ccff', '#1f77b4']
        
        # Calculate percentages
        percentages = []
        for response in likert_order:
            if response in counts.index:
                percentages.append((counts[response] / total) * 100)
            else:
                percentages.append(0)
        
        fig, ax = plt.subplots(figsize=(12, 3))
        
        # Create stacked bar
        left = 0
        for i, (response, pct) in enumerate(zip(likert_order, percentages)):
            if pct > 0:
                bar = ax.barh(0, pct, left=left, color=colors[i], 
                            edgecolor='white', linewidth=2, label=response)
                
                # Add percentage labels
                if pct > 5:  # Only show label if segment is large enough
                    ax.text(left + pct/2, 0, f'{pct:.1f}%',
                           ha='center', va='center', fontsize=10, fontweight='bold')
                
                left += pct
        
        ax.set_xlim(0, 100)
        ax.set_ylim(-0.5, 0.5)
        ax.set_xlabel('Percentage (%)', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_yticks([])
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), 
                 ncol=5, frameon=False)
        
        plt.tight_layout()
        
        return fig
    
    def create_grouped_bar_chart(self,
                                 df: pd.DataFrame,
                                 x_col: str,
                                 y_col: str,
                                 title: str,
                                 x_label: Optional[str] = None,
                                 y_label: Optional[str] = None) -> Figure:
        """
        Create grouped bar chart using matplotlib
        
        Args:
            df: Dataframe with data
            x_col: Column for x-axis
            y_col: Column for y-axis
            title: Chart title
            x_label: Custom x-axis label
            y_label: Custom y-axis label
            
        Returns:
            Matplotlib figure
        """
        agg_df = df.groupby(x_col)[y_col].mean().reset_index()
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        x_pos = np.arange(len(agg_df))
        bars = ax.bar(x_pos, agg_df[y_col], 
                     color=sns.color_palette("Set2", len(agg_df)),
                     edgecolor='black', linewidth=1.2)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=9)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(agg_df[x_col].astype(str), rotation=45, ha='right')
        ax.set_xlabel(x_label or x_col, fontsize=12, fontweight='bold')
        ax.set_ylabel(y_label or y_col, fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        return fig
    
    def create_net_score_bar(self,
                            df: pd.DataFrame,
                            x_col: str,
                            sentiment_col: str,
                            title: str,
                            x_label: Optional[str] = None) -> Figure:
        """
        Create net score bar chart with color coding
        
        Args:
            df: Dataframe with data
            x_col: Column for x-axis
            sentiment_col: Column with sentiment scores
            title: Chart title
            x_label: Custom x-axis label
            
        Returns:
            Matplotlib figure
        """
        clean_df = df[[x_col, sentiment_col]].dropna()
        agg_df = clean_df.groupby(x_col)[sentiment_col].mean().reset_index()
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Color bars based on positive/negative
        colors = ['#2ca02c' if val > 0 else '#d62728' 
                 for val in agg_df[sentiment_col]]
        
        # Ensure categorical x-axis values are strings to avoid dtype issues
        x_labels = agg_df[x_col].astype(str)

        bars = ax.bar(x_labels, agg_df[sentiment_col], 
                     color=colors, edgecolor='black', linewidth=1.2)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            label_y = height + 0.05 if height > 0 else height - 0.05
            va = 'bottom' if height > 0 else 'top'
            ax.text(bar.get_x() + bar.get_width()/2., label_y,
                   f'{height:.2f}',
                   ha='center', va=va, fontsize=9)
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=2)
        ax.set_xlabel(x_label or x_col, fontsize=12, fontweight='bold')
        ax.set_ylabel('Net Score', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3)
        
        # Add legend
        positive_patch = Rectangle((0, 0), 1, 1, fc='#2ca02c', label='Positive')
        negative_patch = Rectangle((0, 0), 1, 1, fc='#d62728', label='Negative')
        ax.legend(handles=[positive_patch, negative_patch])
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        return fig
    
    def create_correlation_heatmap(self,
                                   correlation_matrix: pd.DataFrame,
                                   title: str = "Statement Correlation Matrix") -> Figure:
        """
        Create correlation heatmap using seaborn
        
        Args:
            correlation_matrix: Correlation matrix dataframe
            title: Chart title
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        sns.heatmap(correlation_matrix, 
                   annot=True, 
                   fmt='.2f',
                   cmap='RdBu_r',
                   center=0,
                   square=True,
                   linewidths=1,
                   cbar_kws={"shrink": 0.8},
                   ax=ax)
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        return fig
    
    def create_pie_chart(self,
                        df: pd.DataFrame,
                        column: str,
                        title: str) -> Figure:
        """
        Create pie chart for categorical distribution
        
        Args:
            df: Dataframe with data
            column: Column to visualize
            title: Chart title
            
        Returns:
            Matplotlib figure
        """
        try:
            # Convert to string and drop NaN values
            clean_data = df[column].astype(str).replace(['nan', 'NaN', 'None'], np.nan).dropna()
            counts = clean_data.value_counts()
            
            # Skip if no valid data
            if len(counts) == 0 or counts.sum() == 0:
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.text(0.5, 0.5, 'No data available', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=14)
                ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
                return fig
            
            fig, ax = plt.subplots(figsize=(8, 8))
            
            # Ensure we have valid numeric values
            values = counts.values.astype(float)
            labels = counts.index.astype(str)
            
            colors = sns.color_palette('pastel')[0:len(counts)]
            wedges, texts, autotexts = ax.pie(values, 
                                               labels=labels,
                                               autopct='%1.1f%%',
                                               colors=colors,
                                               startangle=90,
                                               explode=[0.05] * len(counts))
            
            # Beautify text
            for text in texts:
                text.set_fontsize(11)
                text.set_fontweight('bold')
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontsize(10)
                autotext.set_fontweight('bold')
            
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            # Fallback: create error chart
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.text(0.5, 0.5, f'Error creating chart: {str(e)[:50]}...', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
            return fig
    
    def create_box_plot(self,
                       df: pd.DataFrame,
                       x_col: str,
                       y_col: str,
                       title: str) -> Figure:
        """
        Create box plot for distribution analysis
        
        Args:
            df: Dataframe with data
            x_col: Column for x-axis (categories)
            y_col: Column for y-axis (values)
            title: Chart title
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Prepare data for box plot
        categories = df[x_col].unique()
        data_to_plot = [df[df[x_col] == cat][y_col].dropna() for cat in categories]
        
        bp = ax.boxplot(data_to_plot, labels=categories, patch_artist=True,
                       showmeans=True, meanline=True)
        
        # Color the boxes
        colors = sns.color_palette('Set3', len(categories))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_xlabel(x_col, fontsize=12, fontweight='bold')
        ax.set_ylabel(y_col, fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        return fig


def create_visualizer(library: str = 'plotly') -> object:
    """
    Factory function to create appropriate visualizer
    
    Args:
        library: 'plotly' or 'matplotlib'
        
    Returns:
        Visualizer object
    """
    if library.lower() == 'matplotlib':
        return MatplotlibVisualizer()
    else:
        return SurveyVisualizer()