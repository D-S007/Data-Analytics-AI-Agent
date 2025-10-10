import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, Any, Optional, List

class Visualization:
    """Advanced visualization module using Plotly"""
    
    def __init__(self):
        self.color_palette = px.colors.qualitative.Set3
        self.default_template = "plotly_white"
    
    def create_chart(self, data: pd.DataFrame, chart_config: Dict[str, Any]) -> go.Figure:
        """Create a chart based on configuration"""
        
        chart_type = chart_config.get('chart_type', 'bar')
        x_col = chart_config.get('x_column')
        y_col = chart_config.get('y_column')
        color_col = chart_config.get('color_column')
        title = chart_config.get('title', 'Data Visualization')
        
        try:
            if chart_type == 'line':
                fig = self.create_line_chart(data, x_col, y_col, color_col, title)
            elif chart_type == 'bar':
                fig = self.create_bar_chart(data, x_col, y_col, color_col, title)
            elif chart_type == 'scatter':
                fig = self.create_scatter_plot(data, x_col, y_col, color_col, title)
            elif chart_type == 'pie':
                fig = self.create_pie_chart(data, x_col, y_col, title)
            elif chart_type == 'histogram':
                fig = self.create_histogram(data, x_col, title)
            elif chart_type == 'heatmap':
                fig = self.create_heatmap(data, title)
            elif chart_type == 'box':
                fig = self.create_box_plot(data, x_col, y_col, color_col, title)
            else:
                # Fallback to bar chart
                fig = self.create_bar_chart(data, x_col, y_col, color_col, title)
            
            return fig
            
        except Exception as e:
            # Create error visualization
            return self.create_error_chart(str(e))
    
    def create_line_chart(self, data: pd.DataFrame, x_col: str, y_col: str, 
                         color_col: Optional[str] = None, title: str = "Line Chart") -> go.Figure:
        """Create a line chart"""
        
        fig = px.line(
            data, 
            x=x_col, 
            y=y_col, 
            color=color_col,
            title=title,
            template=self.default_template
        )
        
        fig.update_traces(mode='lines+markers')
        fig.update_layout(
            hovermode='x unified',
            xaxis_title=x_col.replace('_', ' ').title() if x_col else 'X-axis',
            yaxis_title=y_col.replace('_', ' ').title() if y_col else 'Y-axis'
        )
        
        return fig
    
    def create_bar_chart(self, data: pd.DataFrame, x_col: str, y_col: Optional[str] = None, 
                        color_col: Optional[str] = None, title: str = "Bar Chart") -> go.Figure:
        """Create a bar chart"""
        
        if y_col is None:
            # Count frequency of x_col values
            value_counts = data[x_col].value_counts().reset_index()
            value_counts.columns = [x_col, 'count']
            
            fig = px.bar(
                value_counts.head(20),  # Limit to top 20 for readability
                x=x_col,
                y='count',
                title=f"{title} - Frequency of {x_col}",
                template=self.default_template
            )
        else:
            fig = px.bar(
                data,
                x=x_col,
                y=y_col,
                color=color_col,
                title=title,
                template=self.default_template
            )
        
        fig.update_layout(
            xaxis_title=x_col.replace('_', ' ').title() if x_col else 'Categories',
            yaxis_title=y_col.replace('_', ' ').title() if y_col else 'Count',
            showlegend=True if color_col else False
        )
        
        return fig
    
    def create_scatter_plot(self, data: pd.DataFrame, x_col: str, y_col: str, 
                           color_col: Optional[str] = None, title: str = "Scatter Plot") -> go.Figure:
        """Create a scatter plot"""
        
        fig = px.scatter(
            data,
            x=x_col,
            y=y_col,
            color=color_col,
            title=title,
            template=self.default_template
        )
        
        fig.update_traces(marker=dict(size=8, opacity=0.7))
        fig.update_layout(
            xaxis_title=x_col.replace('_', ' ').title(),
            yaxis_title=y_col.replace('_', ' ').title(),
            hovermode='closest'
        )
        
        return fig
    
    def create_pie_chart(self, data: pd.DataFrame, x_col: str, y_col: Optional[str] = None, 
                        title: str = "Pie Chart") -> go.Figure:
        """Create a pie chart"""
        
        if y_col is None:
            # Use value counts
            pie_data = data[x_col].value_counts().reset_index()
            pie_data.columns = [x_col, 'count']
            
            fig = px.pie(
                pie_data.head(10),  # Limit to top 10 slices
                values='count',
                names=x_col,
                title=f"{title} - Distribution of {x_col}",
                template=self.default_template
            )
        else:
            # Group by x_col and sum y_col
            pie_data = data.groupby(x_col)[y_col].sum().reset_index()
            
            fig = px.pie(
                pie_data.head(10),
                values=y_col,
                names=x_col,
                title=title,
                template=self.default_template
            )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        
        return fig
    
    def create_histogram(self, data: pd.DataFrame, x_col: str, title: str = "Histogram") -> go.Figure:
        """Create a histogram"""
        
        fig = px.histogram(
            data,
            x=x_col,
            title=f"{title} - Distribution of {x_col}",
            template=self.default_template,
            nbins=30
        )
        
        fig.update_layout(
            xaxis_title=x_col.replace('_', ' ').title(),
            yaxis_title='Frequency',
            bargap=0.1
        )
        
        # Add statistics annotation
        mean_val = data[x_col].mean()
        std_val = data[x_col].std()
        
        fig.add_vline(
            x=mean_val, 
            line_dash="dash", 
            line_color="red",
            annotation_text=f"Mean: {mean_val:.2f}"
        )
        
        return fig
    
    def create_heatmap(self, data: pd.DataFrame, title: str = "Correlation Heatmap") -> go.Figure:
        """Create a correlation heatmap"""
        
        # Get only numeric columns
        numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.empty:
            return self.create_error_chart("No numeric data available for heatmap")
        
        correlation_matrix = numeric_data.corr()
        
        fig = px.imshow(
            correlation_matrix,
            title=title,
            template=self.default_template,
            aspect="auto",
            color_continuous_scale="RdBu_r"
        )
        
        fig.update_layout(
            xaxis_title="Variables",
            yaxis_title="Variables"
        )
        
        return fig
    
    def create_box_plot(self, data: pd.DataFrame, x_col: Optional[str], y_col: str, 
                       color_col: Optional[str] = None, title: str = "Box Plot") -> go.Figure:
        """Create a box plot"""
        
        fig = px.box(
            data,
            x=x_col,
            y=y_col,
            color=color_col,
            title=title,
            template=self.default_template
        )
        
        fig.update_layout(
            xaxis_title=x_col.replace('_', ' ').title() if x_col else 'Category',
            yaxis_title=y_col.replace('_', ' ').title(),
            boxmode='group'
        )
        
        return fig
    
    def create_correlation_heatmap(self, data: pd.DataFrame) -> go.Figure:
        """Create a correlation heatmap for numeric data"""
        
        correlation_matrix = data.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdBu_r',
            zmid=0,
            text=correlation_matrix.round(2).values,
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Correlation Matrix",
            template=self.default_template,
            xaxis_title="Variables",
            yaxis_title="Variables",
            width=600,
            height=600
        )
        
        return fig
    
    def create_distribution_plot(self, data: pd.DataFrame, column: str) -> go.Figure:
        """Create a distribution plot with histogram and box plot"""
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=[f"Distribution of {column}", f"Box Plot of {column}"],
            vertical_spacing=0.12,
            row_heights=[0.7, 0.3]
        )
        
        # Histogram
        fig.add_trace(
            go.Histogram(x=data[column], name="Distribution", nbinsx=30),
            row=1, col=1
        )
        
        # Box plot
        fig.add_trace(
            go.Box(x=data[column], name="Box Plot", orientation='h'),
            row=2, col=1
        )
        
        fig.update_layout(
            title=f"Distribution Analysis: {column}",
            template=self.default_template,
            showlegend=False
        )
        
        return fig
    
    def create_multi_line_chart(self, data: pd.DataFrame, x_col: str, y_cols: List[str], 
                               title: str = "Multi-Line Chart") -> go.Figure:
        """Create a chart with multiple lines"""
        
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set1
        
        for i, y_col in enumerate(y_cols):
            fig.add_trace(go.Scatter(
                x=data[x_col],
                y=data[y_col],
                mode='lines+markers',
                name=y_col.replace('_', ' ').title(),
                line=dict(color=colors[i % len(colors)])
            ))
        
        fig.update_layout(
            title=title,
            template=self.default_template,
            xaxis_title=x_col.replace('_', ' ').title(),
            yaxis_title="Values",
            hovermode='x unified'
        )
        
        return fig
    
    def create_error_chart(self, error_message: str) -> go.Figure:
        """Create an error visualization"""
        
        fig = go.Figure()
        
        fig.add_annotation(
            x=0.5, y=0.5,
            text=f"Visualization Error:<br>{error_message}",
            showarrow=False,
            font=dict(size=16, color="red"),
            align="center"
        )
        
        fig.update_layout(
            title="Visualization Error",
            template=self.default_template,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            width=600,
            height=400
        )
        
        return fig
    
    def create_summary_dashboard(self, data: pd.DataFrame) -> go.Figure:
        """Create a summary dashboard with multiple charts"""
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        
        # Create subplots
        subplot_titles = []
        specs = []
        
        if len(numeric_cols) >= 2:
            subplot_titles.append("Correlation Heatmap")
            specs.append([{"type": "heatmap"}])
        
        if len(numeric_cols) > 0:
            subplot_titles.append(f"Distribution of {numeric_cols[0]}")
            specs.append([{"type": "histogram"}])
        
        if len(categorical_cols) > 0:
            subplot_titles.append(f"Frequency of {categorical_cols[0]}")
            specs.append([{"type": "bar"}])
        
        if not specs:
            return self.create_error_chart("No suitable data for dashboard")
        
        fig = make_subplots(
            rows=len(specs), cols=1,
            subplot_titles=subplot_titles,
            specs=specs,
            vertical_spacing=0.1
        )
        
        row = 1
        
        # Add correlation heatmap
        if len(numeric_cols) >= 2:
            corr_matrix = data[numeric_cols].corr()
            fig.add_trace(
                go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    colorscale='RdBu_r',
                    showscale=False
                ),
                row=row, col=1
            )
            row += 1
        
        # Add histogram
        if len(numeric_cols) > 0:
            fig.add_trace(
                go.Histogram(x=data[numeric_cols[0]], showlegend=False),
                row=row, col=1
            )
            row += 1
        
        # Add bar chart
        if len(categorical_cols) > 0:
            value_counts = data[categorical_cols[0]].value_counts().head(10)
            fig.add_trace(
                go.Bar(x=value_counts.index, y=value_counts.values, showlegend=False),
                row=row, col=1
            )
        
        fig.update_layout(
            title="Data Summary Dashboard",
            template=self.default_template,
            height=300 * len(specs)
        )
        
        return fig
