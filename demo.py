import os
import argparse
import json
from difflib import ndiff
from http.server import HTTPServer, BaseHTTPRequestHandler
import unicodedata

from model import GEC


class S(BaseHTTPRequestHandler):
    def _set_headers(self, content_type='text/html'):
        self.send_response(200)
        self.send_header('Content-type', content_type)
        self.end_headers()

    def do_GET(self):
        if not hasattr(self, 'index_page'):
            dir_path = os.path.dirname(os.path.realpath(__file__))
            index_path = os.path.join(dir_path, 'data/index.html')
            with open(index_path, 'rb') as f:
                self.index_page = f.read()
        self._set_headers()
        self.wfile.write(self.index_page)

    def do_HEAD(self):
        self._set_headers()

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        text = json.loads(post_data.decode())['text'][:512]
        text = unicodedata.normalize('NFKC', text).replace(' ', '')
        correct_text = gec.correct(text)
        diffs = list(ndiff(text, correct_text))
        print(f'Correction: {text} -> {correct_text}')
        output_dict = {
            'correctedText': correct_text,
            'diffs': diffs
        }
        out_bytes = json.dumps(output_dict, separators=(',',':')).encode()
        self._set_headers(content_type='application/json')
        self.wfile.write(out_bytes)


def run(server_class=HTTPServer, handler_class=S, addr='localhost', port=8000):
    server_address = (addr, port)
    httpd = server_class(server_address, handler_class)

    print(f'Starting httpd server on {addr}:{port}')
    httpd.serve_forever()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a simple HTTP server')
    parser.add_argument('-l', '--listen', default='localhost',
        help='Specify the IP address on which the server listens')
    parser.add_argument('-p', '--port', type=int, default=8000,
        help='Specify the port on which the server listens')
    parser.add_argument('-w', '--weights', required=True,
        help='Path to model weights')
    args = parser.parse_args()
    gec = GEC(pretrained_weights_path=args.weights)
    run(addr=args.listen, port=args.port)
