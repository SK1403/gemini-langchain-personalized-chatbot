#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
##
# Description:-
#
#       Google Cloud Platform authentication and credential verification utilities.
#       Discovers and validates active GCP Project IDs and Application Default Credentials (ADC).
#
##
# Development date    Developed by       Comments
# ----------------    ------------       ---------
# 04/10/2024          Saddam Khan        Initial implementation
# 22/10/2024          Saddam Khan        Added GCP project and ADC credential resolution
#
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import os
import google.auth
from google.auth.exceptions import DefaultCredentialsError

def resolve_gcp_project(configured_project: str = "") -> str:
    """
    Explanation: Resolves GCP project ID by checking explicit parameter, environment variables, and ADC
    :param  configured_project str: User-specified GCP project identifier
    :return project_id str: Resolved GCP project ID string or empty string
    """
    if configured_project and configured_project.strip():
        return configured_project.strip()

    env_project = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCLOUD_PROJECT")
    if env_project and env_project.strip():
        return env_project.strip()

    try:
        credentials, project_id = google.auth.default()
        if project_id:
            return project_id
    except DefaultCredentialsError:
        pass

    return ""

def verify_adc() -> tuple[bool, str]:
    """
    Explanation: Verifies whether Google Application Default Credentials (ADC) are configured
    :return is_valid bool: True if credentials are valid, False otherwise
    :return details str: Project ID or error traceback string
    """
    try:
        credentials, project_id = google.auth.default()
        return True, project_id or ""
    except DefaultCredentialsError as e:
        return False, str(e)
