// Add shim to support Jupyter 3.x and 4.x
var Jupyter = Jupyter || IPython || {};
Jupyter.notebook = Jupyter.notebook || {};
var isInitDone = false;
const AUTO_EXEC_FLAG = "!AUTO_EXEC";
const AUTO_OUT_FLAG = "!AUTO_OUT";
var groups = ['required_args', 'optional_args', 'returns'];
var groupLabels = ['Input', 'Optional Input', 'Output'];

/**
 * Wait for kernel before initializing extension.
 */
var initWrapper = function() {
  var interval = setInterval(function() {
    if (Jupyter.notebook.kernel && Jupyter.notebook.kernel.is_connected() && !Jupyter.notebook.kernel_busy) {
      init();
      clearInterval(interval);
    }
  }, 50);
};

/**
 * Initializes the extension.
 */
var init = function() {
  setupCallbacks();
  autoRunWidgets();
  addMenuOptions();
  mapKeyboardShortcuts();

  console.log('Initialized Simpli nbextension.');
};

/**
 * Setup cell execution callbacks to the notebook kernel.
 */
var setupCallbacks = function() {
  var initCode = `
from inspect import getsource
from simpli import startup
exec(getsource(startup))
  `;

  Jupyter.notebook.insert_cell_at_index('code', 0);
  var cell = Jupyter.notebook.get_cell(0);
  cell.set_text(initCode);
  cell.execute();
  Jupyter.notebook.delete_cell(0);

  // TODO: Initialize extension on kernel restart
  console.log('Called setupCallbacks()');
}

/**
 * Automatically run all Simpli widgets on initialization.
 */
var autoRunWidgets = function() {
  console.log('Called autoRunWidgets()');
  $.each($(".cell"), function(index, value) {
    var cellCode = $(value).html();
    if (cellCode.indexOf(AUTO_EXEC_FLAG) > -1) {
      toSimpliCell(index);
    }
  });
};

/**
 * Add menu options to notebook navbar and toolbar.
 */
var addMenuOptions = function() {

  // Add button for creating Simpli cell to toolbar
  Jupyter.toolbar.add_buttons_group([
    {
      'label': 'Insert Simpli Cell',
      'icon': 'fa-bolt', // select from http://fortawesome.github.io/Font-Awesome/icons/
      'callback': function() {
        Jupyter.notebook.insert_cell_below();
        Jupyter.notebook.select_next();
        showTaskList();
      }
    }
  ]);

  // Add button for converting code to Simpli Widget
  Jupyter.toolbar.add_buttons_group([
    {
      'label': 'Simpli Widget <-> Code',
      'icon': 'fa-exchange', // select from http://fortawesome.github.io/Font-Awesome/icons/
      'callback': function() {
        var cellIndex = Jupyter.notebook.get_selected_index();
        var cell = Jupyter.notebook.get_selected_cell();
        var cell_text = cell.get_text();

        // Convert widget to code
        if (cell_text.indexOf(AUTO_EXEC_FLAG) > -1) {
          var pythonTask = JSON.stringify(getWidgetData(cell));
          var code = `manager.code_task('''${pythonTask}''')`;
          console.log(pythonTask);

          var setCode = function(out) {
            console.log(out);
            cell.set_text(out.content.text.trim());
            cell.clear_output();
            showCellInput(cell);
          }

          Jupyter.notebook.kernel.execute(code, {
            'iopub': {
              'output': setCode
            }
          });
        } else {
          // Convert code to widget
          var code = `manager.get_task(notebook_cell_text='''${cell_text}''')`;

          var toSimpliCellWrap = function(out) {
            toSimpliCell(null, out);
          }
          getTask(null, cell_text, toSimpliCellWrap);
        }
      }
    }
  ]);

  // Initialize the undo delete menu entry click function
  var undeleteCell = $('#undelete_cell a');
  undeleteCell.on("click", function(event) {
    undoDeleteCell();
  });
};

/**
 * Initialize custom keyboard shortcuts for Simpli.
 */
var mapKeyboardShortcuts = function() {
  // Initialize the Simpli cell type keyboard shortcut
  Jupyter.keyboard_manager.command_shortcuts.add_shortcut('shift-x', {
    help: 'to Simpli',
    handler: function() {
      showTaskList();
      return false;
    }
  });

  // Initialize the undo delete keyboard shortcut
  Jupyter.keyboard_manager.command_shortcuts.add_shortcut('z', {
    help: 'undo cell/widget deletion',
    handler: function() {
      undoDeleteCell();
      return false;
    }
  });

  // Handle esc key
  $('body').keydown(function(event) {

    // Remove focus from active element
    if (event.keyCode == 27 && event.shiftKey) {
      document.activeElement.blur();
    }

    // Close the library
    if (event.keyCode == 27 && $('#library-right-panel-close').length) {
      $('#library-right-panel-close').click();
      return;
    }

    // Select current task
    if (event.keyCode == 13 && $('#library-select-btn').length) {
      $('#library-select-btn').click();
    }
  });
}

/**
 * Undo deleting last set of cells/widgets.
 * FIXME: test if it actually works
 */
var undoDeleteCell = function() {
  // Make sure there are deleted cells to restore
  if (Jupyter.notebook.undelete_backup == null)
    return;

  var backup = Jupyter.notebook.undelete_backup;
  var startIndex = Jupyter.notebook.undelete_index;
  var endIndex = startIndex + backup.length;
  var indices = _.range(startIndex, endIndex);

  // Reinsert deleted cells appropriately
  Jupyter.notebook.undelete_cell();
  for (var i in indices) {
    var cell = $(".cell")[i];
    if ($(cell).html().indexOf(AUTO_EXEC_FLAG) >= 0) {
      toSimpliCell(i);
    }
  }
};

/**
 * Show the input and prompt for the specified notebook cell.
 * @param  {Number} cell Notebook cell to be hidden
 */
var showCellInput = function(cell) {
  cell.element.removeClass("simpli-cell");
}

/**
 * Converts indicated cell to Simpli widget and hiding code input.
 * @param  {number} index    index of cell to convert
 * @param  {Object} taskJSON task JSON object
 */
var toSimpliCell = function(index, taskJSON) {
  // Use index if provided. Otherwise use index of currently selected cell.
  if (index == null) {
    index = Jupyter.notebook.get_selected_index();
  }

  cell = Jupyter.notebook.get_cell(index);

  // Wait for kernel to not be busy
  var interval = setInterval(function() {
    if (!Jupyter.notebook.kernel_busy) {
      clearInterval(interval);

      // Force cell type to code
      var cell_type = cell.cell_type;
      if (cell_type !== "code") {
        Jupyter.notebook.to_code(index);
      }

      if (taskJSON == undefined) {
        renderTaskWidget(index);
      } else {
        renderTaskWidget(index, taskJSON);
      }
    }
  }, 10);

};

var STATIC_PATH = location.origin + Jupyter.contents.base_url + "nbextensions/simpli/resources/";

define([
    'base/js/namespace',
    'base/js/events',
    'jquery',
    STATIC_PATH + 'taskList.js',
    STATIC_PATH + 'taskWidget.js'
], function(Jupyter, events) {
  function load_ipython_extension() {
    // Inject custom CSS
    $('head')
      .append(
        $('<link />')
        .attr('rel', 'stylesheet')
        .attr('type', 'text/css')
        .attr('href', STATIC_PATH + 'theme.css')
      );

    // Wait for the kernel to be ready and then initialize the widgets
    initWrapper();
  }

  return {
    load_ipython_extension: load_ipython_extension
  };
});
