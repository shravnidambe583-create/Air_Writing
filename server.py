import http.server
import socketserver
import json
import base64
import os
import time
from urllib.parse import urlparse

import sys

PORT = 8080
if len(sys.argv) > 1:
    try:
        PORT = int(sys.argv[1])
    except ValueError:
        print(f"[WARNING] Invalid port provided. Using default port: {PORT}")

DIRECTORY = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(DIRECTORY, "saved_drawings")

class AirWritingHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        # Serve from the current directory
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def do_POST(self):
        parsed_path = urlparse(self.path)
        if parsed_path.path == '/api/save':
            try:
                # Get Content-Length header
                content_length = int(self.headers.get('Content-Length', 0))
                if content_length == 0:
                    self.send_error_response(400, "Missing Content-Length")
                    return

                # Read body
                post_data = self.rfile.read(content_length)
                # Check Content-Type to handle raw binary or JSON base64
                content_type = self.headers.get('Content-Type', '')
                if 'application/json' in content_type:
                    data = json.loads(post_data.decode('utf-8'))
                    image_data = data.get('image')
                    if not image_data or not image_data.startswith("data:image/png;base64,"):
                        self.send_error_response(400, "Invalid image data format")
                        return
                    header, encoded = image_data.split(",", 1)
                    decoded = base64.b64decode(encoded)
                elif 'image/png' in content_type:
                    decoded = post_data
                else:
                    self.send_error_response(400, f"Unsupported Content-Type: {content_type}")
                    return
                
                # Ensure directory exists
                os.makedirs(SAVE_DIR, exist_ok=True)
                
                # Create timestamped filename
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"drawing_{timestamp}.png"
                filepath = os.path.join(SAVE_DIR, filename)
                
                # Write file
                with open(filepath, "wb") as f:
                    f.write(decoded)
                
                print(f"[SUCCESS] Saved drawing: {filepath}")

                # Send response
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                response = {
                    'status': 'success', 
                    'filename': filename,
                    'filepath': os.path.relpath(filepath, DIRECTORY)
                }
                self.wfile.write(json.dumps(response).encode('utf-8'))
            except Exception as e:
                print(f"[ERROR] Failed to save drawing: {e}")
                self.send_error_response(500, f"Server Error: {str(e)}")
        else:
            self.send_error_response(404, "Not Found")

    def do_OPTIONS(self):
        # Support CORS preflight requests if needed
        self.send_response(204)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def send_error_response(self, code, message):
        self.send_response(code)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        response = {'status': 'error', 'message': message}
        self.wfile.write(json.dumps(response).encode('utf-8'))

def run_server():
    # Make sure we change port if it's already in use
    handler = AirWritingHandler
    
    # Try binding to port 8000, fallback to other ports if occupied
    port = PORT
    server = None
    for attempt in range(5):
        try:
            server = socketserver.TCPServer(("", port), handler)
            break
        except OSError:
            print(f"[INFO] Port {port} is occupied. Trying next port...")
            port += 1
            
    if not server:
        print("[ERROR] Could not find an open port to bind the server.")
        return

    print("\n" + "="*50)
    print(f" Viya Local Server Started Successfully!")
    print(f" Serving folder: {DIRECTORY}")
    print(f" Open your browser and navigate to:")
    print(f" --> http://localhost:{port}")
    print("="*50 + "\n")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[INFO] Server shutting down...")
        server.server_close()
        print("[INFO] Server stopped.")

if __name__ == "__main__":
    run_server()
