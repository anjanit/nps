class DataService:
    def __init__(self, db_session):
        self.db_session = db_session

    def fetch_data_by_scheme(self, scheme_name):
        result = self.db_session.query(Schemas).filter(Schemas.scheme_name == scheme_name).all()
        return result