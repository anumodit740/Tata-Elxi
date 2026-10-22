"""One command: integrity checks, then the actual Streamlit model app."""
from pathlib import Path
import subprocess,sys,argparse
ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8501);args=parser.parse_args()
    subprocess.run([sys.executable,'-B',str(ROOT/'verify_package.py')],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'-B','-m','streamlit','run','app.py',f'--server.port={args.port}','--server.address=127.0.0.1','--server.headless=true','--browser.gatherUsageStats=false'],cwd=ROOT,check=True)
