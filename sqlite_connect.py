import sqlite3  # Import the SQLite library for database operations
from pathlib import Path  # Import Path from pathlib for handling file paths

# Define the path to the SQLite database file
# The database file is located in a "data" folder relative to the script's location
DB_PATH = Path(__file__).resolve().parent / "data/data.db"

def init_connection():
    """
    Initialize a connection to the SQLite database.

    This function creates a connection to the SQLite database specified by `DB_PATH`.
    It also creates a cursor object to execute SQL queries and enables autocommit mode.

    Returns:
        tuple: A tuple containing the connection object and the cursor object.
    """
    # Establish a connection to the SQLite database
    conn = sqlite3.connect(DB_PATH)
    
    # Create a cursor object to execute SQL commands
    cursor = conn.cursor()
    
    # Enable autocommit mode to automatically save changes to the database
    conn.autocommit = True

    # Return both the connection and cursor objects
    return conn, cursor

if __name__ == "__main__":
    # Initialize the database connection and cursor
    conn, cursor = init_connection()
    
    # Example query: Retrieve the SQLite version
    cursor.execute("SELECT sqlite_version();")  # Execute a query to get the SQLite version
    data = cursor.fetchone()  # Fetch the first row of the result
    
    # Print the SQLite version to the console
    print(f"SQLite version: {data[0]}")
    
    # Close the database connection to release resources
    conn.close()