import json

import mysql.connector

from datetime import datetime

from app.core.config import (
    MYSQL_HOST,
    MYSQL_PORT,
    MYSQL_USER,
    MYSQL_PASSWORD,
    MYSQL_DATABASE
)


# ==========================================
# MYSQL CONNECTION
# ==========================================

def get_connection():

    return mysql.connector.connect(

        host=MYSQL_HOST,

        port=MYSQL_PORT,

        user=MYSQL_USER,

        password=MYSQL_PASSWORD,

        database=MYSQL_DATABASE
    )


# ==========================================
# CREATE TABLE
# ==========================================

def create_table():

    connection = get_connection()

    cursor = connection.cursor()


    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (

            request_id VARCHAR(100) PRIMARY KEY,

            timestamp DATETIME NOT NULL,

            model_used VARCHAR(20) NOT NULL,

            input_features JSON NOT NULL,

            prediction INT NOT NULL,

            probability DECIMAL(10,8) NOT NULL,

            actual_outcome INT NULL

        )
    """)


    connection.commit()

    cursor.close()

    connection.close()


# ==========================================
# LOG PREDICTION
# ==========================================

def log_prediction(

    request_id,

    model_used,

    input_features,

    prediction,

    probability

):

    connection = get_connection()

    cursor = connection.cursor()


    timestamp = datetime.now()


    query = """

        INSERT INTO predictions (

            request_id,

            timestamp,

            model_used,

            input_features,

            prediction,

            probability,

            actual_outcome

        )

        VALUES (

            %s,

            %s,

            %s,

            %s,

            %s,

            %s,

            %s

        )

    """


    values = (

        request_id,

        timestamp,

        model_used,

        json.dumps(input_features),

        prediction,

        probability,

        None

    )


    cursor.execute(

        query,

        values

    )


    connection.commit()

    cursor.close()

    connection.close()


# ==========================================
# UPDATE FEEDBACK
# ==========================================

def update_actual_outcome(

    request_id,

    actual_outcome

):

    connection = get_connection()

    cursor = connection.cursor()


    query = """

        UPDATE predictions

        SET actual_outcome = %s

        WHERE request_id = %s

    """


    cursor.execute(

        query,

        (

            actual_outcome,

            request_id

        )

    )


    rows_updated = cursor.rowcount


    connection.commit()

    cursor.close()

    connection.close()


    return rows_updated