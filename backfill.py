import database as db

if __name__ == "__main__":
    print("--- Database Backfill Utility ---")
    print("This script will populate the database with historical stock data.")
    print("This may take several minutes to complete.")

    # 1. Initialize the database and create tables
    db.initialize_db()

    # 2. Run the backfill process
    db.backfill_all_tickers()

    print("--- Backfill Complete ---")
    print("The database 'history.db' is now populated.")