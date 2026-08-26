REQUIRED_FIELDS = [
    "age",
    "gender",
    "income",
    "loan_amount",
    "credit_score",
    "existing_loans",
    "employment_years",
    "education",
    "self_employed",
    "loan_term"
]


ALLOWED_GENDER = [
    "Male",
    "Female"
]


ALLOWED_EDUCATION = [
    "High School",
    "Bachelor",
    "Master",
    "PhD"
]


ALLOWED_SELF_EMPLOYED = [
    "Yes",
    "No"
]


def validate_prediction_request(data):

    errors = {}

    # ======================================
    # REQUIRED FIELDS
    # ======================================

    for field in REQUIRED_FIELDS:

        if field not in data:

            errors[field] = (
                "This field is required."
            )


    if errors:

        return errors


    # ======================================
    # AGE
    # ======================================

    if not isinstance(data["age"], int):

        errors["age"] = (
            "Age must be an integer."
        )

    elif data["age"] <= 18 or data["age"] >= 100:

        errors["age"] = (
            "Age must be between 19 and 99."
        )


    # ======================================
    # GENDER
    # ======================================

    if data["gender"] not in ALLOWED_GENDER:

        errors["gender"] = (
            "Gender must be Male or Female."
        )


    # ======================================
    # INCOME
    # ======================================

    if (
        not isinstance(
            data["income"],
            (int, float)
        )
        or data["income"] <= 0
    ):

        errors["income"] = (
            "Income must be greater than 0."
        )


    # ======================================
    # LOAN AMOUNT
    # ======================================

    if (
        not isinstance(
            data["loan_amount"],
            (int, float)
        )
        or data["loan_amount"] <= 0
    ):

        errors["loan_amount"] = (
            "Loan amount must be greater than 0."
        )


    # ======================================
    # CREDIT SCORE
    # ======================================

    if not isinstance(
        data["credit_score"],
        int
    ):

        errors["credit_score"] = (
            "Credit score must be an integer."
        )

    elif not (
        300 <= data["credit_score"] <= 850
    ):

        errors["credit_score"] = (
            "Credit score must be between 300 and 850."
        )


    # ======================================
    # EXISTING LOANS
    # ======================================

    if (
        not isinstance(
            data["existing_loans"],
            int
        )
        or data["existing_loans"] < 0
    ):

        errors["existing_loans"] = (
            "Existing loans must be 0 or greater."
        )


    # ======================================
    # EMPLOYMENT YEARS
    # ======================================

    if (
        not isinstance(
            data["employment_years"],
            int
        )
        or data["employment_years"] < 0
    ):

        errors["employment_years"] = (
            "Employment years must be 0 or greater."
        )


    # ======================================
    # EDUCATION
    # ======================================

    if data["education"] not in ALLOWED_EDUCATION:

        errors["education"] = (
            "Invalid education value."
        )


    # ======================================
    # SELF EMPLOYED
    # ======================================

    if (
        data["self_employed"]
        not in ALLOWED_SELF_EMPLOYED
    ):

        errors["self_employed"] = (
            "self_employed must be Yes or No."
        )


    # ======================================
    # LOAN TERM
    # ======================================

    if (
        not isinstance(
            data["loan_term"],
            int
        )
        or data["loan_term"] <= 0
    ):

        errors["loan_term"] = (
            "Loan term must be greater than 0."
        )


    return errors