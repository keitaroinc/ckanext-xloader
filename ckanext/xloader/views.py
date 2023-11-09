from flask import Blueprint

from ckanapi import LocalCKAN
from ckan.plugins.toolkit import _, h, g, render, request, abort, NotAuthorized

import ckanext.xloader.utils as utils


xloader = Blueprint("xloader", __name__)


def get_blueprints():
    return [xloader]


@xloader.route("/dataset/<id>/resource_data/<resource_id>", methods=("GET", "POST"))
def resource_data(id, resource_id):
    rows = request.args.get('rows')
    if rows:
        try:
            rows = int(rows)
            if rows < 0:
                rows = None
        except ValueError:
            rows = None
    return utils.resource_data(id, resource_id, rows)


@xloader.route("/dataset/<id>/delete-datastore/<resource_id>", methods=("GET", "POST"))
def delete_datastore_table(id, resource_id):
    if u'cancel' in request.form:
        return h.redirect_to(u'xloader.resource_data', id=id, resource_id=resource_id)

    if request.method == 'POST':
        lc = LocalCKAN(username=g.user)

        try:
            lc.action.datastore_delete(
                resource_id=resource_id,
                force=True,
            )
        except NotAuthorized:
            return abort(403, _(u'Unauthorized to delete resource %s') % resource_id)

        h.flash_notice(_(u'DataStore and Data Dictionary deleted for resource %s') % resource_id)

        return h.redirect_to(
            'xloader.resource_data',
            id=id,
            resource_id=resource_id
        )
    else:
        g.resource_id = resource_id
        g.package_id = id

        extra_vars = {
            u"resource_id": resource_id,
            u"package_id": id
        }
        return render(u'xloader/confirm_datastore_delete.html', extra_vars)
