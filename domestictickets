import streamlit as st
import pandas as pd
import datetime
import uuid
from datetime import timedelta
import plotly.express as px
import plotly.graph_objects as go
import json
import sqlite3
import os
from pathlib import Path

# Configure page
st.set_page_config(
    page_title="Household Chore Tracker",
    page_icon="🏠",
    layout="wide"
)

# Data storage configuration
DATA_DIR = Path("chore_data")
DATA_DIR.mkdir(exist_ok=True)

CSV_FILE = DATA_DIR / "chores.csv"
JSON_FILE = DATA_DIR / "chores.json"
SQLITE_FILE = DATA_DIR / "chores.db"
CONFIG_FILE = DATA_DIR / "config.json"

class DataManager:
    """Handles data persistence with multiple storage options"""
    
    def __init__(self, storage_type="json"):
        self.storage_type = storage_type
        self.setup_storage()
    
    def setup_storage(self):
        """Initialize storage based on type"""
        if self.storage_type == "sqlite":
            self.setup_sqlite()
    
    def setup_sqlite(self):
        """Create SQLite tables if they don't exist"""
        conn = sqlite3.connect(SQLITE_FILE)
        cursor = conn.cursor()
        
        # Create chores table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chores (
                submission_id TEXT PRIMARY KEY,
                item_name TEXT NOT NULL,
                deadline DATE,
                priority TEXT,
                estimated_time INTEGER,
                assigned_to TEXT,
                created_by TEXT,
                created_date DATE,
                status TEXT,
                completed_date DATE,
                priority_score REAL
            )
        ''')
        
        # Create users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                created_date DATE
            )
        ''')
        
        # Create config table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS config (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_chores(self, chores):
        """Save chores to persistent storage"""
        if self.storage_type == "csv":
            self.save_to_csv(chores)
        elif self.storage_type == "json":
            self.save_to_json(chores)
        elif self.storage_type == "sqlite":
            self.save_to_sqlite(chores)
    
    def load_chores(self):
        """Load chores from persistent storage"""
        if self.storage_type == "csv":
            return self.load_from_csv()
        elif self.storage_type == "json":
            return self.load_from_json()
        elif self.storage_type == "sqlite":
            return self.load_from_sqlite()
        return []
    
    def save_to_csv(self, chores):
        """Save chores to CSV file"""
        if chores:
            df = pd.DataFrame(chores)
            # Convert date objects to strings for CSV storage
            date_columns = ['deadline', 'created_date', 'completed_date']
            for col in date_columns:
                if col in df.columns:
                    df[col] = df[col].astype(str)
            df.to_csv(CSV_FILE, index=False)
    
    def load_from_csv(self):
        """Load chores from CSV file"""
        if CSV_FILE.exists():
            try:
                df = pd.read_csv(CSV_FILE)
                # Convert string dates back to date objects
                date_columns = ['deadline', 'created_date', 'completed_date']
                for col in date_columns:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col], errors='coerce').dt.date
                
                # Replace NaT with None
                df = df.where(pd.notnull(df), None)
                return df.to_dict('records')
            except Exception as e:
                st.error(f"Error loading CSV: {e}")
        return []
    
    def save_to_json(self, chores):
        """Save chores to JSON file"""
        # Convert date objects to strings for JSON serialization
        chores_for_json = []
        for chore in chores:
            chore_copy = chore.copy()
            for key, value in chore_copy.items():
                if isinstance(value, datetime.date):
                    chore_copy[key] = value.isoformat() if value else None
            chores_for_json.append(chore_copy)
        
        with open(JSON_FILE, 'w') as f:
            json.dump(chores_for_json, f, indent=2)
    
    def load_from_json(self):
        """Load chores from JSON file"""
        if JSON_FILE.exists():
            try:
                with open(JSON_FILE, 'r') as f:
                    chores = json.load(f)
                
                # Convert string dates back to date objects
                for chore in chores:
                    date_fields = ['deadline', 'created_date', 'completed_date']
                    for field in date_fields:
                        if field in chore and chore[field]:
                            try:
                                chore[field] = datetime.datetime.fromisoformat(chore[field]).date()
                            except:
                                chore[field] = None
                
                return chores
            except Exception as e:
                st.error(f"Error loading JSON: {e}")
        return []
    
    def save_to_sqlite(self, chores):
        """Save chores to SQLite database"""
        conn = sqlite3.connect(SQLITE_FILE)
        cursor = conn.cursor()
        
        # Clear existing data
        cursor.execute('DELETE FROM chores')
        
        # Insert new data
        for chore in chores:
            cursor.execute('''
                INSERT INTO chores VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                chore['submission_id'],
                chore['item_name'],
                chore['deadline'].isoformat() if chore['deadline'] else None,
                chore['priority'],
                chore['estimated_time'],
                chore['assigned_to'],
                chore['created_by'],
                chore['created_date'].isoformat() if chore['created_date'] else None,
                chore['status'],
                chore['completed_date'].isoformat() if chore['completed_date'] else None,
                chore['priority_score']
            ))
        
        conn.commit()
        conn.close()
    
    def load_from_sqlite(self):
        """Load chores from SQLite database"""
        if SQLITE_FILE.exists():
            try:
                conn = sqlite3.connect(SQLITE_FILE)
                cursor = conn.cursor()
                
                cursor.execute('SELECT * FROM chores')
                rows = cursor.fetchall()
                
                chores = []
                for row in rows:
                    chore = {
                        'submission_id': row[0],
                        'item_name': row[1],
                        'deadline': datetime.datetime.fromisoformat(row[2]).date() if row[2] else None,
                        'priority': row[3],
                        'estimated_time': row[4],
                        'assigned_to': row[5],
                        'created_by': row[6],
                        'created_date': datetime.datetime.fromisoformat(row[7]).date() if row[7] else None,
                        'status': row[8],
                        'completed_date': datetime.datetime.fromisoformat(row[9]).date() if row[9] else None,
                        'priority_score': row[10]
                    }
                    chores.append(chore)
                
                conn.close()
                return chores
            except Exception as e:
                st.error(f"Error loading SQLite: {e}")
        return []
    
    def save_config(self, config):
        """Save configuration data"""
        with open(CONFIG_FILE, 'w') as f:
            # Convert any date objects to strings
            config_for_json = {}
            for key, value in config.items():
                if isinstance(value, datetime.date):
                    config_for_json[key] = value.isoformat()
                else:
                    config_for_json[key] = value
            json.dump(config_for_json, f, indent=2)
    
    def load_config(self):
        """Load configuration data"""
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, 'r') as f:
                    return json.load(f)
            except Exception as e:
                st.error(f"Error loading config: {e}")
        return {}

# Initialize data manager
if 'data_manager' not in st.session_state:
    st.session_state.data_manager = DataManager("json")  # Default to JSON

# Initialize session state
if 'initialized' not in st.session_state:
    # Load configuration
    config = st.session_state.data_manager.load_config()
    
    # Initialize with saved config or defaults
    st.session_state.users = config.get('users', ['User 1', 'User 2'])
    st.session_state.current_user = config.get('current_user', 'User 1')
    st.session_state.priority_weights = config.get('priority_weights', {
        'Low': 1,
        'Medium': 2,
        'High': 3,
        'Urgent': 5
    })
    
    # Load chores from persistent storage
    st.session_state.chores = st.session_state.data_manager.load_chores()
    st.session_state.initialized = True

def save_data():
    """Save all data to persistent storage"""
    # Save chores
    st.session_state.data_manager.save_chores(st.session_state.chores)
    
    # Save configuration
    config = {
        'users': st.session_state.users,
        'current_user': st.session_state.current_user,
        'priority_weights': st.session_state.priority_weights,
        'storage_type': st.session_state.data_manager.storage_type
    }
    st.session_state.data_manager.save_config(config)

def backup_data():
    """Create a backup of all data"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = DATA_DIR / "backups"
    backup_dir.mkdir(exist_ok=True)
    
    # Always create JSON backup for portability
    backup_file = backup_dir / f"chores_backup_{timestamp}.json"
    
    backup_data = {
        'chores': st.session_state.chores,
        'users': st.session_state.users,
        'priority_weights': st.session_state.priority_weights,
        'backup_timestamp': datetime.datetime.now().isoformat()
    }
    
    # Convert dates for JSON serialization
    chores_for_json = []
    for chore in backup_data['chores']:
        chore_copy = chore.copy()
        for key, value in chore_copy.items():
            if isinstance(value, datetime.date):
                chore_copy[key] = value.isoformat() if value else None
        chores_for_json.append(chore_copy)
    
    backup_data['chores'] = chores_for_json
    
    with open(backup_file, 'w') as f:
        json.dump(backup_data, f, indent=2)
    
    return backup_file

def calculate_priority_score(priority, deadline, estimated_time):
    """Calculate dynamic priority score based on multiple factors"""
    base_score = st.session_state.priority_weights[priority]
    
    # Add urgency based on deadline
    if deadline:
        days_until = (deadline - datetime.date.today()).days
        if days_until <= 0:
            urgency_multiplier = 3.0  # Overdue
        elif days_until <= 1:
            urgency_multiplier = 2.5  # Due today/tomorrow
        elif days_until <= 3:
            urgency_multiplier = 2.0  # Due within 3 days
        elif days_until <= 7:
            urgency_multiplier = 1.5  # Due within a week
        else:
            urgency_multiplier = 1.0  # Future
    else:
        urgency_multiplier = 1.0
    
    # Factor in estimated time (longer tasks get slight priority boost)
    time_factor = 1 + (estimated_time / 480)  # 480 minutes = 8 hours max boost
    
    return base_score * urgency_multiplier * time_factor

def add_chore(item_name, deadline, priority, estimated_time, assigned_to):
    """Add a new chore to the system"""
    submission_id = str(uuid.uuid4())[:8].upper()
    
    chore = {
        'submission_id': submission_id,
        'item_name': item_name,
        'deadline': deadline,
        'priority': priority,
        'estimated_time': estimated_time,
        'assigned_to': assigned_to,
        'created_by': st.session_state.current_user,
        'created_date': datetime.date.today(),
        'status': 'Pending',
        'completed_date': None,
        'priority_score': calculate_priority_score(priority, deadline, estimated_time)
    }
    
    st.session_state.chores.append(chore)
    save_data()  # Auto-save when adding chore
    return submission_id

def update_chore_status(submission_id, new_status):
    """Update the status of a chore"""
    for chore in st.session_state.chores:
        if chore['submission_id'] == submission_id:
            chore['status'] = new_status
            if new_status == 'Completed':
                chore['completed_date'] = datetime.date.today()
            else:
                chore['completed_date'] = None
            break
    save_data()  # Auto-save when updating status

def recalculate_all_priorities():
    """Recalculate priority scores for all pending chores"""
    for chore in st.session_state.chores:
        if chore['status'] == 'Pending':
            chore['priority_score'] = calculate_priority_score(
                chore['priority'], chore['deadline'], chore['estimated_time']
            )
    save_data()  # Auto-save when recalculating

# Sidebar for data management and user settings
st.sidebar.header("💾 Data Management")

# Storage type selection
storage_options = ["JSON (Recommended)", "CSV", "SQLite Database"]
storage_mapping = {"JSON (Recommended)": "json", "CSV": "csv", "SQLite Database": "sqlite"}

current_storage_display = next(k for k, v in storage_mapping.items() 
                              if v == st.session_state.data_manager.storage_type)

selected_storage = st.sidebar.selectbox(
    "Storage Type:",
    options=storage_options,
    index=storage_options.index(current_storage_display)
)

new_storage_type = storage_mapping[selected_storage]
if new_storage_type != st.session_state.data_manager.storage_type:
    st.session_state.data_manager = DataManager(new_storage_type)
    save_data()  # Save with new storage type

# Data operations
col1, col2 = st.sidebar.columns(2)
with col1:
    if st.button("💾 Save Now"):
        save_data()
        st.success("Data saved!")

with col2:
    if st.button("📋 Backup"):
        backup_file = backup_data()
        st.success(f"Backup created!")
        st.caption(f"Saved to: {backup_file.name}")

# Show data location
with st.sidebar.expander("📁 Data Location"):
    st.code(f"Data folder: {DATA_DIR.absolute()}")
    st.write("Files:")
    for file_path in [CSV_FILE, JSON_FILE, SQLITE_FILE, CONFIG_FILE]:
        if file_path.exists():
            size = file_path.stat().st_size
            st.write(f"✅ {file_path.name} ({size} bytes)")
        else:
            st.write(f"❌ {file_path.name}")

# Export options
with st.sidebar.expander("📤 Export Data"):
    if st.session_state.chores:
        # Export as CSV for Excel
        df = pd.DataFrame(st.session_state.chores)
        csv_data = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name=f"chores_export_{datetime.date.today()}.csv",
            mime="text/csv"
        )
        
        # Export as JSON
        json_data = json.dumps(st.session_state.chores, indent=2, default=str)
        st.download_button(
            label="Download JSON",
            data=json_data,
            file_name=f"chores_export_{datetime.date.today()}.json",
            mime="application/json"
        )

st.sidebar.divider()

# User management
st.sidebar.header("👤 User Management")

# Current user selection
current_user = st.sidebar.selectbox(
    "Select Current User:",
    st.session_state.users,
    index=st.session_state.users.index(st.session_state.current_user)
)

if current_user != st.session_state.current_user:
    st.session_state.current_user = current_user
    save_data()

# Add new user
with st.sidebar.expander("Add New User"):
    new_user = st.text_input("New User Name:")
    if st.button("Add User") and new_user:
        if new_user not in st.session_state.users:
            st.session_state.users.append(new_user)
            save_data()
            st.success(f"Added user: {new_user}")
            st.rerun()
        else:
            st.warning("User already exists!")

# Priority weight configuration
st.sidebar.header("⚖️ Priority Configuration")
with st.sidebar.expander("Adjust Priority Weights"):
    st.write("Configure how priority levels are weighted:")
    weights_changed = False
    
    for priority in ['Low', 'Medium', 'High', 'Urgent']:
        old_weight = st.session_state.priority_weights[priority]
        new_weight = st.slider(
            f"{priority} Weight:",
            min_value=1,
            max_value=10,
            value=old_weight,
            key=f"weight_{priority}"
        )
        if new_weight != old_weight:
            st.session_state.priority_weights[priority] = new_weight
            weights_changed = True
    
    if weights_changed:
        save_data()
    
    if st.button("Recalculate All Priorities"):
        recalculate_all_priorities()
        st.success("Priority scores updated!")
        st.rerun()

# Main application
st.title("🏠 Household Chore Tracker")
st.write(f"**Current User:** {st.session_state.current_user}")

# Data persistence indicator
storage_indicator = {
    "json": "📄 JSON",
    "csv": "📊 CSV", 
    "sqlite": "🗄️ SQLite"
}
st.caption(f"💾 Data Storage: {storage_indicator[st.session_state.data_manager.storage_type]} | Auto-saves all changes")

# Create tabs for different views
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📝 Add New Chore", 
    "📋 All Chores", 
    "⏰ My Tasks", 
    "📊 Analytics", 
    "🔧 Manage Chores"
])

with tab1:
    st.header("Add New Chore")
    
    col1, col2 = st.columns(2)
    
    with col1:
        item_name = st.text_input("Chore Name*:", placeholder="e.g., Vacuum living room")
        deadline = st.date_input(
            "Deadline:",
            value=datetime.date.today() + timedelta(days=7),
            min_value=datetime.date.today()
        )
        priority = st.select_slider(
            "Priority Level:",
            options=['Low', 'Medium', 'High', 'Urgent'],
            value='Medium'
        )
    
    with col2:
        estimated_time = st.number_input(
            "Estimated Time (minutes):",
            min_value=5,
            max_value=480,
            value=30,
            step=5
        )
        assigned_to = st.selectbox(
            "Assign to:",
            st.session_state.users
        )
        
        # Show calculated priority score preview
        if item_name:
            preview_score = calculate_priority_score(priority, deadline, estimated_time)
            st.metric("Priority Score Preview", f"{preview_score:.1f}")
    
    if st.button("Create Chore", type="primary"):
        if item_name:
            submission_id = add_chore(item_name, deadline, priority, estimated_time, assigned_to)
            st.success(f"✅ Chore created and saved! Tracking ID: **{submission_id}**")
            st.rerun()
        else:
            st.error("Please enter a chore name!")

with tab2:
    st.header("All Chores")
    
    if st.session_state.chores:
        # Filter options
        col1, col2, col3 = st.columns(3)
        with col1:
            status_filter = st.multiselect(
                "Filter by Status:",
                options=['Pending', 'In Progress', 'Completed'],
                default=['Pending', 'In Progress']
            )
        with col2:
            user_filter = st.multiselect(
                "Filter by Assigned User:",
                options=st.session_state.users,
                default=st.session_state.users
            )
        with col3:
            priority_filter = st.multiselect(
                "Filter by Priority:",
                options=['Low', 'Medium', 'High', 'Urgent'],
                default=['Low', 'Medium', 'High', 'Urgent']
            )
        
        # Create DataFrame for display
        df = pd.DataFrame(st.session_state.chores)
        
        # Apply filters
        filtered_df = df[
            (df['status'].isin(status_filter)) &
            (df['assigned_to'].isin(user_filter)) &
            (df['priority'].isin(priority_filter))
        ]
        
        if not filtered_df.empty:
            # Sort by priority score (descending) and deadline
            filtered_df = filtered_df.sort_values(['priority_score', 'deadline'], ascending=[False, True])
            
            # Display chores
            for _, chore in filtered_df.iterrows():
                with st.container():
                    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                    
                    with col1:
                        # Status indicator
                        status_color = {
                            'Pending': '🔴',
                            'In Progress': '🟡',
                            'Completed': '🟢'
                        }
                        
                        st.write(f"**{chore['item_name']}** {status_color[chore['status']]}")
                        st.caption(f"ID: {chore['submission_id']} | Assigned to: {chore['assigned_to']} | Created by: {chore['created_by']}")
                    
                    with col2:
                        st.write(f"**Priority:** {chore['priority']}")
                        st.caption(f"Score: {chore['priority_score']:.1f}")
                    
                    with col3:
                        st.write(f"**Deadline:** {chore['deadline']}")
                        days_left = (chore['deadline'] - datetime.date.today()).days
                        if days_left < 0:
                            st.caption(f"⚠️ {abs(days_left)} days overdue")
                        elif days_left == 0:
                            st.caption("📅 Due today")
                        else:
                            st.caption(f"📅 {days_left} days left")
                    
                    with col4:
                        st.write(f"**Time:** {chore['estimated_time']} min")
                        st.write(f"**Status:** {chore['status']}")
                
                st.divider()
        else:
            st.info("No chores match the current filters.")
    else:
        st.info("No chores created yet. Add your first chore in the 'Add New Chore' tab!")

with tab3:
    st.header(f"My Tasks - {st.session_state.current_user}")
    
    if st.session_state.chores:
        # Get tasks assigned to current user
        my_chores = [chore for chore in st.session_state.chores 
                    if chore['assigned_to'] == st.session_state.current_user]
        
        if my_chores:
            # Separate by status
            pending_chores = [c for c in my_chores if c['status'] == 'Pending']
            in_progress_chores = [c for c in my_chores if c['status'] == 'In Progress']
            completed_chores = [c for c in my_chores if c['status'] == 'Completed']
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Tasks", len(my_chores))
            with col2:
                st.metric("Pending", len(pending_chores))
            with col3:
                st.metric("In Progress", len(in_progress_chores))
            with col4:
                st.metric("Completed", len(completed_chores))
            
            # Task management
            st.subheader("📋 Active Tasks")
            active_chores = pending_chores + in_progress_chores
            
            if active_chores:
                # Sort by priority score
                active_chores.sort(key=lambda x: x['priority_score'], reverse=True)
                
                for chore in active_chores:
                    with st.container():
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.write(f"**{chore['item_name']}**")
                            st.caption(f"Priority: {chore['priority']} (Score: {chore['priority_score']:.1f}) | "
                                     f"Deadline: {chore['deadline']} | Time: {chore['estimated_time']} min")
                        
                        with col2:
                            new_status = st.selectbox(
                                "Status:",
                                options=['Pending', 'In Progress', 'Completed'],
                                index=['Pending', 'In Progress', 'Completed'].index(chore['status']),
                                key=f"status_{chore['submission_id']}"
                            )
                            
                            if new_status != chore['status']:
                                update_chore_status(chore['submission_id'], new_status)
                                st.success(f"Status updated and saved!")
                                st.rerun()
                    
                    st.divider()
            else:
                st.info("🎉 No active tasks! Great job!")
            
            # Recent completions
            if completed_chores:
                st.subheader("✅ Recent Completions")
                recent_completed = sorted(completed_chores, 
                                        key=lambda x: x['completed_date'], 
                                        reverse=True)[:5]
                
                for chore in recent_completed:
                    st.write(f"✅ **{chore['item_name']}** - Completed on {chore['completed_date']}")
        else:
            st.info(f"No tasks assigned to {st.session_state.current_user} yet.")
    else:
        st.info("No chores created yet.")

with tab4:
    st.header("📊 Analytics & Insights")
    
    if st.session_state.chores:
        df = pd.DataFrame(st.session_state.chores)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Chores by status
            status_counts = df['status'].value_counts()
            fig_status = px.pie(
                values=status_counts.values,
                names=status_counts.index,
                title="Chores by Status",
                color_discrete_map={
                    'Pending': '#ff6b6b',
                    'In Progress': '#feca57',
                    'Completed': '#48dbfb'
                }
            )
            st.plotly_chart(fig_status, use_container_width=True)
            
            # Priority distribution
            priority_counts = df['priority'].value_counts()
            fig_priority = px.bar(
                x=priority_counts.index,
                y=priority_counts.values,
                title="Chores by Priority Level",
                color=priority_counts.index,
                color_discrete_map={
                    'Low': '#95e1d3',
                    'Medium': '#f3d250',
                    'High': '#f38ba8',
                    'Urgent': '#e74c3c'
                }
            )
            st.plotly_chart(fig_priority, use_container_width=True)
        
        with col2:
            # Workload by user
            user_counts = df['assigned_to'].value_counts()
            fig_users = px.bar(
                x=user_counts.index,
                y=user_counts.values,
                title="Tasks by User",
                color=user_counts.index
            )
            st.plotly_chart(fig_users, use_container_width=True)
            
            # Completion rate by user
            completion_rates = []
            for user in st.session_state.users:
                user_chores = df[df['assigned_to'] == user]
                if len(user_chores) > 0:
                    completed = len(user_chores[user_chores['status'] == 'Completed'])
                    rate = (completed / len(user_chores)) * 100
                    completion_rates.append({'User': user, 'Completion Rate': rate})
            
            if completion_rates:
                completion_df = pd.DataFrame(completion_rates)
                fig_completion = px.bar(
                    completion_df,
                    x='User',
                    y='Completion Rate',
                    title="Completion Rate by User (%)",
                    color='Completion Rate',
                    color_continuous_scale='RdYlGn'
                )
                st.plotly_chart(fig_completion, use_container_width=True)
        
        # Time series analysis
        st.subheader("📈 Completion Trends")
        if len(df[df['status'] == 'Completed']) > 0:
            completed_df = df[df['status'] == 'Completed'].copy()
            completed_df['completed_date'] = pd.to_datetime(completed_df['completed_date'])
            completed_df['week'] = completed_df['completed_date'].dt.to_period('W')
            
            weekly_completions = completed_df.groupby('week').size().reset_index(name='completions')
            weekly_completions['week'] = weekly_completions['week'].astype(str)
            
            fig_trend = px.line(
                weekly_completions,
                x='week',
                y='completions',
                title="Weekly Completion Trend",
                markers=True
            )
            fig_trend.update_xaxes(title="Week")
            fig_trend.update_yaxes(title="Completed Tasks")
            st.plotly_chart(fig_trend, use_container_width=True)
        
        # Summary statistics
        st.subheader("📈 Summary Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_time = df['estimated_time'].mean()
            st.metric("Avg. Task Time", f"{avg_time:.0f} min")
        
        with col2:
            overdue_tasks = len(df[(df['status'] != 'Completed') & 
                                 (df['deadline'] < datetime.date.today())])
            st.metric("Overdue Tasks", overdue_tasks)
        
        with col3:
            total_pending_time = df[df['status'] == 'Pending']['estimated_time'].sum()
            st.metric("Pending Work", f"{total_pending_time} min")
        
        with col4:
            high_priority = len(df[df['priority'].isin(['High', 'Urgent'])])
            st.metric("High Priority Tasks", high_priority)
        
        # Historical data insights
        if len(df) > 10:
            st.subheader("📊 Historical Insights")
            col1, col2 = st.columns(2)
            
            with col1:
                # Average completion time by priority
                completed_with_time = df[df['status'] == 'Completed']
                if len(completed_with_time) > 0:
                    avg_by_priority = completed_with_time.groupby('priority')['estimated_time'].mean().reset_index()
                    fig_priority_time = px.bar(
                        avg_by_priority,
                        x='priority',
                        y='estimated_time',
                        title="Average Task Duration by Priority",
                        color='priority'
                    )
                    st.plotly_chart(fig_priority_time, use_container_width=True)
            
            with col2:
                # Monthly task creation
                df['created_month'] = pd.to_datetime(df['created_date']).dt.to_period('M')
                monthly_created = df.groupby('created_month').size().reset_index(name='tasks_created')
                monthly_created['created_month'] = monthly_created['created_month'].astype(str)
                
                fig_monthly = px.bar(
                    monthly_created,
                    x='created_month',
                    y='tasks_created',
                    title="Tasks Created by Month",
                    color='tasks_created',
                    color_continuous_scale='Blues'
                )
                st.plotly_chart(fig_monthly, use_container_width=True)
    
    else:
        st.info("No data available for analytics yet. Create some chores to see insights!")

with tab5:
    st.header("🔧 Manage Chores")
    
    if st.session_state.chores:
        st.subheader("Search and Edit")
        
        # Search functionality
        search_term = st.text_input("🔍 Search chores by name or ID:")
        
        # Filter chores based on search
        filtered_chores = st.session_state.chores
        if search_term:
            filtered_chores = [
                chore for chore in st.session_state.chores
                if search_term.lower() in chore['item_name'].lower() or 
                   search_term.upper() in chore['submission_id']
            ]
        
        if filtered_chores:
            # Select chore to edit
            chore_options = {
                f"{chore['submission_id']} - {chore['item_name']}": chore
                for chore in filtered_chores
            }
            
            selected_chore_key = st.selectbox(
                "Select chore to manage:",
                options=list(chore_options.keys())
            )
            
            if selected_chore_key:
                selected_chore = chore_options[selected_chore_key]
                
                st.subheader(f"Editing: {selected_chore['item_name']}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Edit form
                    new_name = st.text_input("Chore Name:", value=selected_chore['item_name'])
                    new_deadline = st.date_input("Deadline:", value=selected_chore['deadline'])
                    new_priority = st.select_slider(
                        "Priority:",
                        options=['Low', 'Medium', 'High', 'Urgent'],
                        value=selected_chore['priority']
                    )
                    new_time = st.number_input(
                        "Estimated Time (min):",
                        value=selected_chore['estimated_time'],
                        min_value=5,
                        max_value=480
                    )
                    new_assigned = st.selectbox(
                        "Assigned to:",
                        options=st.session_state.users,
                        index=st.session_state.users.index(selected_chore['assigned_to'])
                    )
                    new_status = st.selectbox(
                        "Status:",
                        options=['Pending', 'In Progress', 'Completed'],
                        index=['Pending', 'In Progress', 'Completed'].index(selected_chore['status'])
                    )
                
                with col2:
                    st.write("**Current Details:**")
                    st.write(f"ID: {selected_chore['submission_id']}")
                    st.write(f"Created by: {selected_chore['created_by']}")
                    st.write(f"Created on: {selected_chore['created_date']}")
                    st.write(f"Current priority score: {selected_chore['priority_score']:.1f}")
                    if selected_chore['completed_date']:
                        st.write(f"Completed on: {selected_chore['completed_date']}")
                
                # Action buttons
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("💾 Save Changes", type="primary"):
                        # Update the chore
                        for chore in st.session_state.chores:
                            if chore['submission_id'] == selected_chore['submission_id']:
                                chore['item_name'] = new_name
                                chore['deadline'] = new_deadline
                                chore['priority'] = new_priority
                                chore['estimated_time'] = new_time
                                chore['assigned_to'] = new_assigned
                                chore['status'] = new_status
                                if new_status == 'Completed' and chore['completed_date'] is None:
                                    chore['completed_date'] = datetime.date.today()
                                elif new_status != 'Completed':
                                    chore['completed_date'] = None
                                chore['priority_score'] = calculate_priority_score(new_priority, new_deadline, new_time)
                                break
                        
                        save_data()  # Save changes
                        st.success("✅ Chore updated and saved successfully!")
                        st.rerun()
                
                with col2:
                    if st.button("🗑️ Delete Chore", type="secondary"):
                        st.session_state.chores = [
                            chore for chore in st.session_state.chores
                            if chore['submission_id'] != selected_chore['submission_id']
                        ]
                        save_data()  # Save after deletion
                        st.success("🗑️ Chore deleted and saved successfully!")
                        st.rerun()
        else:
            st.info("No chores found matching your search.")
        
        # Bulk operations
        st.subheader("🔄 Bulk Operations")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Recalculate All Priorities"):
                recalculate_all_priorities()
                st.success("All priority scores recalculated and saved!")
                st.rerun()
        
        with col2:
            if st.button("🧹 Clear Completed Tasks"):
                initial_count = len(st.session_state.chores)
                st.session_state.chores = [
                    chore for chore in st.session_state.chores
                    if chore['status'] != 'Completed'
                ]
                removed_count = initial_count - len(st.session_state.chores)
                if removed_count > 0:
                    save_data()  # Save after bulk deletion
                    st.success(f"Removed {removed_count} completed tasks and saved!")
                    st.rerun()
                else:
                    st.info("No completed tasks to remove.")
        
        with col3:
            if st.button("🔄 Reset All Data"):
                if st.checkbox("⚠️ I understand this will delete ALL data"):
                    st.session_state.chores = []
                    save_data()
                    st.success("All data cleared and saved!")
                    st.rerun()
        
        # Data import section
        st.subheader("📥 Import Data")
        uploaded_file = st.file_uploader(
            "Upload backup file (JSON format):",
            type=['json'],
            help="Upload a previously exported JSON backup file"
        )
        
        if uploaded_file is not None:
            try:
                import_data = json.load(uploaded_file)
                
                # Validate the structure
                if 'chores' in import_data:
                    # Show preview
                    st.write("**Preview of import data:**")
                    st.write(f"- {len(import_data['chores'])} chores")
                    if 'users' in import_data:
                        st.write(f"- Users: {', '.join(import_data['users'])}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("📥 Import (Replace All)", type="primary"):
                            # Convert date strings back to date objects
                            imported_chores = []
                            for chore in import_data['chores']:
                                chore_copy = chore.copy()
                                date_fields = ['deadline', 'created_date', 'completed_date']
                                for field in date_fields:
                                    if field in chore_copy and chore_copy[field]:
                                        try:
                                            chore_copy[field] = datetime.datetime.fromisoformat(chore_copy[field]).date()
                                        except:
                                            chore_copy[field] = None
                                imported_chores.append(chore_copy)
                            
                            st.session_state.chores = imported_chores
                            
                            if 'users' in import_data:
                                st.session_state.users = import_data['users']
                            if 'priority_weights' in import_data:
                                st.session_state.priority_weights = import_data['priority_weights']
                            
                            save_data()
                            st.success("Data imported and saved successfully!")
                            st.rerun()
                    
                    with col2:
                        if st.button("📥 Import (Append)", type="secondary"):
                            # Convert and append
                            for chore in import_data['chores']:
                                chore_copy = chore.copy()
                                # Ensure unique IDs
                                chore_copy['submission_id'] = str(uuid.uuid4())[:8].upper()
                                
                                date_fields = ['deadline', 'created_date', 'completed_date']
                                for field in date_fields:
                                    if field in chore_copy and chore_copy[field]:
                                        try:
                                            chore_copy[field] = datetime.datetime.fromisoformat(chore_copy[field]).date()
                                        except:
                                            chore_copy[field] = None
                                
                                st.session_state.chores.append(chore_copy)
                            
                            save_data()
                            st.success(f"Appended {len(import_data['chores'])} chores and saved!")
                            st.rerun()
                else:
                    st.error("Invalid file format. Please upload a valid JSON backup file.")
            
            except Exception as e:
                st.error(f"Error importing file: {e}")
    
    else:
        st.info("No chores to manage yet.")

# Footer with data persistence information
st.divider()
col1, col2 = st.columns([2, 1])
with col1:
    st.caption("🏠 Household Chore Tracker - Keep your home organized!")
with col2:
    if st.session_state.chores:
        st.caption(f"💾 {len(st.session_state.chores)} chores stored persistently")
    else:
        st.caption("💾 Ready to store your chores persistently")
