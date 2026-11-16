import unittest

from v13_parallel_collection import bounded_map, source_access_stop


MISSING_ORIGINAL = ('404 Client Error: Not Found for url: '
    'https://nsearchives.nseindia.com/corporate/xbrl/'
    'INTEGRATED_FILING_INDAS_1500429_01082025060000_WEB.xml')


class HTTPStatusStopTests(unittest.TestCase):
    def test_real_missing_original_does_not_abandon_remaining_jobs(self):
        result = list(bounded_map(range(8), lambda n:
            {'error': MISSING_ORIGINAL} if n == 0 else {'item': n}, workers=1))
        self.assertEqual(len(result), 8)
        self.assertEqual([item for item, _ in result], list(range(8)))

    def test_url_digits_and_path_words_are_not_access_stop_reasons(self):
        for error in [MISSING_ORIGINAL,
            '404 Client Error: Not Found for url: https://example.test/403/forbidden/429.xml',
            'Read timed out for https://example.test/429.xml',
            'Missing or ambiguous original identity 1500429']:
            with self.subTest(error=error):
                self.assertFalse(source_access_stop({'error': error}))

    def test_actual_http_access_stops_still_abandon_the_queue(self):
        for error in ['429 Client Error: Too Many Requests for url: https://example.test/404.xml',
                      'HTTP Error 403: Forbidden', 'HTTP/2 429 Too Many Requests']:
            with self.subTest(error=error):
                result = list(bounded_map(range(8), lambda n: {'error': error}, workers=1))
                self.assertEqual(len(result), 1)

    def test_structured_response_status_takes_precedence_over_resource_text(self):
        self.assertFalse(source_access_stop({'http_status': 404, 'error': 'Forbidden filename'}))
        self.assertTrue(source_access_stop({'http_status': 429, 'error': 'Missing resource'}))
        self.assertTrue(source_access_stop({'status_code': '403'}))
        self.assertFalse(source_access_stop({'status_code': '4290', 'error': 'Missing identity'}))
        self.assertFalse(source_access_stop({'http_status': 200}))

    def test_explicit_provider_stop_reasons_without_http_status_are_preserved(self):
        for error in ['Too Many Requests', 'Forbidden', 'Error: forbidden',
                      "HTTPSConnectionPool: ResponseError('too many 429 error responses')"]:
            with self.subTest(error=error):
                self.assertTrue(source_access_stop({'error': error}))

    def test_health_uses_the_same_transport_status_decision(self):
        from v13_collection import _rate_limit_observed
        self.assertFalse(_rate_limit_observed([{'error': MISSING_ORIGINAL}]))
        self.assertTrue(_rate_limit_observed([{'error': 'HTTP 429 Too Many Requests'}]))
        self.assertTrue(_rate_limit_observed([{'http_status': 403}]))


if __name__ == '__main__':
    unittest.main()
