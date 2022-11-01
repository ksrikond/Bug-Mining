"""
Data Extraction Queries for HSD Mining Analysis Application

This file contains all the fields related to HSD query.

QUERY_PARAMS: String of comma-separated fields. These fields are basic fields, hence, they do not need 'server.bugeco.' appended before them
EXTRA_PARAMS_LIST: List of advanced HSD fields which need <tenant_name>.<subject_name>. as a prefix. For example, 'server.bugeco.', 'sighting_central.sighting.' etc.
EXTRA_PARAMS_STR: String of EXTRA_PARAMS_LIST after appending the prefix
PAYLOAD_BY_*: Query payloads by different conditions
QUERY_DICT: Dictionary to store the payload by it's name for config file
"""
import src.utils.hsd_util as _util

QUERY_PARAMS = "id, title, description, status, priority, reason, family, release, component, release_affected, " \
               "component_affected, tag, owner, eta, eta_request, closed_by, closed_date, updated_by, updated_date," \
               " submitted_by,submitted_date, tenant_affected, nickname, notify, comments"

EXTRA_PARAMS_LIST = ['type', 'collateral_type_server', 'gk_cluster', 'gk_stepping', 'gk_branch', 'generation',
                     'ip_domain', 'ip_generation', 'local_change_only', 'partition_affected', 'dss_affected_products',
                     'release_found', 'gate', 'build_found', 'team_found', 'env_found', 'test_found', 'dut_found',
                     'requesting_project', 'scratchpad', 'results_directory', 'to_reproduce', 'failure_signature',
                     'ar_gen', 'workaround_status', 'workaround_id', 'root_cause', 'por', 'ccb_template', 'open_date',
                     'approved_date', 'defined_date', 'repo_modified_date', 'ccb_status', 'ccb_mode', 'ccb_prq_gating',
                     'ccb_order', 'ccb_workaround', 'ccb_por', 'ccb_meeting', 'ccb_disposition_type', 'ccb_driver_id',
                     'errata_status', 'errata_info_owner', 'drop_found', 'drop_fix_plan', 'drop_fix', 'retro_status',
                     'retro_owner', 'retro_ip_escape', 'retro_analysis', 'retro_learnings']

EXTRA_PARAMS_STR = _util.append_before(EXTRA_PARAMS_LIST, "server.bugeco.")

PAYLOAD_BY_COMPONENT = """
{ 
    "eql": "select """ + QUERY_PARAMS + ", " + EXTRA_PARAMS_STR + """ where component contains 'ip.top'"
}"""


PAYLOAD_BY_OWNER = """
{ 
    "eql": "select """ + QUERY_PARAMS + ", " + EXTRA_PARAMS_STR + """ where owner contains 'ksrikond'" 
}"""

PAYLOAD_TEST = """{ "eql": "select """ + QUERY_PARAMS + """ where server.bugeco.id = 14011346840" }"""

QUERY_DICT = {
    'PAYLOAD_BY_COMPONENT': PAYLOAD_BY_COMPONENT,
    'PAYLOAD_BY_OWNER': PAYLOAD_BY_OWNER,
    'PAYLOAD_TEST': PAYLOAD_TEST
}
