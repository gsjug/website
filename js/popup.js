/**
 * Created by mpredli01 on 5/19/16.
 */

var windowObjectReference = null; // global variable

function openRequestedPopup(strUrl,strWindowName) {
    if(windowObjectReference == null || windowObjectReference.closed) {
        windowObjectReference = window.open(strUrl,strWindowName,"width=500,height=500,resizable,scrollbars=yes,status=1");
        }
    else {
        windowObjectReference.focus();
        };
    }
