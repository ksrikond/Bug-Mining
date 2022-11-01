class HsdDataObject(object):
    """
    HSD Sighting object - HSDDataObject maps all the fields from a HSD Document
    """
    def __init__(self, id, title, description, status, priority, reason, family, release,
                 component, release_affected, component_affected, tag, owner, eta,
                 eta_request, closed_by, closed_date, updated_by, updated_date,
                 submitted_by, submitted_date, tenant_affected, nickname, notify, comments,
                 *args, **kwargs):
        self.id = id
        self.title = title
        self.description = description
        self.status = status
        self.priority = priority
        self.reason = reason
        self.family = family
        self.release = release
        self.component = component
        self.release_affected = release_affected
        self.component_affected = component_affected
        self.tag = tag
        self.owner = owner
        self.eta = eta
        self.eta_request = eta_request
        self.closed_by = closed_by
        self.closed_date = closed_date
        self.updated_by = updated_by
        self.updated_date = updated_date
        self.submitted_by = submitted_by
        self.submitted_date = submitted_date
        self.tenant_affected = tenant_affected
        self.nickname = nickname
        self.notify = notify
        self.comments = comments

