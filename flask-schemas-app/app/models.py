from app import db

class Scheme(db.Model):
    __tablename__ = 'schemes'
    
    scheme_code = db.Column(db.String(20), primary_key=True)
    scheme_name = db.Column(db.Text, nullable=False)

class Historical(db.Model):
    __tablename__ = 'historical'
    
    scheme_code = db.Column(db.String(20), db.ForeignKey('schemes.scheme_code'), primary_key=True)
    e_date = db.Column(db.Date, primary_key=True)
    nav = db.Column(db.Float, nullable=False)