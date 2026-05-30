from sqlalchemy import create_engine, text

# Conectar ao PostgreSQL
engine = create_engine("postgresql://leonardoxavier:123456@localhost/healthcare_db")

# Testar conexão
with engine.connect() as conn:
    result = conn.execute(text("SELECT 1"))
    print("Conexão bem-sucedida! Resultado:", result.scalar())
