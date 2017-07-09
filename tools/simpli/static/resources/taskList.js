// Add shim to support Jupyter 3.x and 4.x
var Jupyter = Jupyter || IPython || {}
Jupyter.notebook = Jupyter.notebook || {};
const STATIC_LIB_PATH = location.origin + Jupyter.contents.base_url + "nbextensions/simpli/resources/";

/**
 * Label (UID) of selected task in reference to simpliTaskData.
 * @type {String}
 */
var selectedLabel;

/**
 * Inner container o formatf dialog for task selection.
 */
var taskListParent;

/**
 * Panel that displays selected task information.
 */
var infoPanel;

/**
 * Panel that lists all tasks detailed in simpliTaskData.
 */
var leftPanel;

/******************** Manager Interface Functions ********************/

var getTasks = function(callback) {
  // code to read library JSON files
  var code = `manager.get_tasks()`;

  // Convert tasks JSON to stringified list
  var my_callback = function(out) {
    console.log(out);
    var tasksDict = JSON.parse(out.content.text);
    tasks = Object.keys(tasksDict).map(function(key) {
      var task = tasksDict[key];
      task.label = key;
      return task;
    });
    return tasks;
  }

  var allCallbacks = function(out) {
    var tasks = my_callback(out);
    callback(tasks);
  }

  // Wait for kernel to not be busy
  var interval = setInterval(function() {
    // Use kernel to read library JSONs
    if (!Jupyter.notebook.kernel_busy) {
      clearInterval(interval);
      Jupyter.notebook.kernel.execute(code, {
        'iopub': {
          'output': allCallbacks
        }
      });
    }
  }, 10);
}

/**
 * Request a task from the Simpli manager.
 * @param  {String}   taskLabel UID label for the task.
 * @param  {String}   notebook_cell_text UID label for the task.
 * @param  {Function} callback  Function performed after request finishes.
 */
var getTask = function(taskLabel, notebook_cell_text, callback) {
  // code to retrieve json from Simpli manager

  var code;
  if (taskLabel != null) {
    code = `manager.get_task(task_label='''${taskLabel}''')`;
  } else if (notebook_cell_text != null) {
    code = `manager.get_task(notebook_cell_text='''${notebook_cell_text}''')`;
  } else {
    throw "Need either task_label or notebook_cell_text.";
  }

  // Convert stringified task JSON to JSON object
  var my_callback = function(out) {
    var task = JSON.parse(out.content.text);
    return task;
  }

  var allCallbacks = function(out) {
    var task = my_callback(out);
    callback(task);
  }

  // Wait for kernel to not be busy
  var interval = setInterval(function() {
    // Use kernel to read library JSONs
    if (!Jupyter.notebook.kernel_busy) {
      clearInterval(interval);
      Jupyter.notebook.kernel.execute(code, {
        'iopub': {
          'output': allCallbacks
        }
      });
    }
  }, 10);
}

/******************** MAIN FUNCTIONS ********************/
/**
 * Creates dialog modal that user can select a task from.
 */
var showTaskList = function() {
  initTaskList();

  var dialog = require('base/js/dialog');
  dialog.modal({
    notebook: Jupyter.notebook,
    keyboard_manager: Jupyter.notebook.keyboard_manager,
    body: taskListParent
  });

  // Style parent after it renders
  var interval = setInterval(function() {
    if ($('#library-parent').length > 0) {
      var libParent = $('#library-parent');
      libParent.parent().addClass('library-modal-body');
      libParent.parents('.modal-content').find('.modal-header').addClass('library-modal-header');
      libParent.parents('.modal-content').find('.modal-footer').addClass('library-modal-footer');
      libParent.parents('.modal-dialog').addClass('library-modal-dialog').on('click', function(event) {
        event.preventDefault();
      });
      clearInterval(interval);
    }
  }, 100);
}

/**
 * Initialize panels inside task dialog and saves to taskListParent object.
 */
var initTaskList = function() {
  taskListParent = $('<div/>').attr('id', 'library-parent');

  // Display tasks elements
  leftPanel = $('<div/>')
    .addClass('library-left-panel')
    .addClass('pull-left')
    .addClass('col-xs-7')
    .appendTo(taskListParent);

  var leftPanelHeader = $('<h1/>')
    .addClass('library-left-panel-header')
    .html('Simpli Tasks')
    .appendTo(leftPanel);

  // Specifically to hold cards
  var leftPanelInner = $('<div/>')
    .addClass('library-left-panel-inner')
    .appendTo(leftPanel);

  // Define right panel
  infoPanel = $('<div/>')
    .attr('id', 'library-right-panel')
    .addClass('pull-right')
    .addClass('col-xs-5')
    .appendTo(taskListParent);

  var closeIcon = $('<div/>')
    .attr('id', 'library-right-panel-close')
    .attr('data-dismiss', 'modal')
    .html('CLOSE')
    .appendTo(infoPanel);
  // .click(function() {
  // });
  //
  getTasks(function(tasks) {
    renderTasks(tasks);
    renderInfoPanel();
  });
}

/**
 * Create left panel showing list of tasks.
 * @param  {Object} tasks JSON object representing tasks.
 */
var renderTasks = function(tasks) {
  console.log('Called renderTasks()');

  var loadText = $('<div/>')
    .addClass('library-load-text')
    .html('Loading...')
    .appendTo(leftPanel);

  // Sort tasks by package then task label alphabetically
  tasks.sort(function(a, b) {
    var alib = a.library_name.toLowerCase();
    var blib = b.library_name.toLowerCase();
    var alab = a.label.toLowerCase();
    var blab = b.label.toLowerCase();

    // Sort by package
    if (alib > blib) {
      return 1;
    } else if (alib == blib) {
      // Sort by task label
      if (alab > blab) {
        return 1;
      } else if (alab < blab) {
        return -1;
      } else {
        return 0;
      }
    } else {
      return -1;
    }
  })

  // Hide loading text
  $(leftPanel).find('.library-load-text').addClass('library-load-text-hidden');

  // Render all tasks after loading text fades
  setTimeout(function() {
    var packages = {};
    for (var task of tasks) {
      var tasklib = task.library_name.toUpperCase();

      // Section headers = package names
      if (!(tasklib in packages)) {
        packages[tasklib] = 0;
        var packageHeader = $('<h3/>')
          .addClass('library-package-header')
          .html(tasklib);
        $(leftPanel).find('.library-left-panel-inner').append(packageHeader);
      }
      renderTask(task);
    }
  }, 200);
}

/**
 * Render info panel and only updates inner content when necessary.
 * @param  {Object} task JSON object representing task.
 */
var renderInfoPanel = function(task) {
  // Render right panel
  var render = function() {

    // Parent container
    var taskInfo = $('<div/>')
      .attr('id', 'library-task-info');

    // Task title
    var taskHeading = $('<h2/>')
      .attr('id', 'library-task-heading')
      .appendTo(taskInfo);

    // Task library name
    var taskLibraryName = $('<h3/>')
      .attr('id', 'library-task-package')
      .appendTo(taskInfo);

    // Package author
    var taskAuthor = $('<div/>')
      .attr('id', 'library-task-author')
      .appendTo(taskInfo);

    // Task description
    var taskDescription = $('<div/>')
      .attr('id', 'library-task-description')
      .appendTo(taskInfo);

    // Select/cancel buttons
    var modalButtons = $('<div/>')
      .attr('id', 'library-button-group');
    var selectButton = $('<button>')
      .addClass('library-select-btn')
      .addClass('btn')
      .addClass('btn-default')
      .addClass('btn-primary')
      .attr('data-dismiss', 'modal')
      .html('Select')
      .on('click', function(event) {
        event.preventDefault();
        getTask(selectedLabel, null, function(selectedTask) {
          toSimpliCell(Jupyter.notebook.get_selected_index(), selectedTask);
        });
      })
      .appendTo(modalButtons);
    var cancelButton = $('<button>')
      .attr('id', 'library-cancel-btn')
      .addClass('btn')
      .addClass('btn-default')
      .attr('data-dismiss', 'modal')
      .html('Cancel')
      .appendTo(modalButtons);

    taskInfo.appendTo(infoPanel);
    modalButtons.appendTo(infoPanel);
  };

  /**
   * Update existing infoPanel with currently selected task
   */
  var update = function() {
    // Parse and display task information
    getTask(selectedLabel, null, function(task) {
      var label = Object.keys(task)[0];
      task = task[label];
      $(infoPanel).find('#library-task-heading').html(label);
      $(infoPanel).find('#library-task-package').html(task.library_name);
      $(infoPanel).find('#library-task-author').html(task.author);
      $(infoPanel).find('#library-task-description').html(task.description);
    });

  }

  // Render if first call. Otherwise update with selected task data
  if (infoPanel.children().length == 1) {
    render();
  } else {
    update();
  }
}

/**
 * Render a card for a given task JSON string. Also responsible for triggering right panel display.
 * @param {String} task_data stringified JSON for a task
 */
var renderTask = function(task) {

  // Generate a card from given task_data
  var card = $('<a/>')
    .addClass('library-card')
    .on('click', function(event) {
      event.preventDefault();

      // click action
      selectedLabel = $(this).find('h4').html();
      getTask(selectedLabel, null, function(selectedTask) {
        renderInfoPanel(task);
      });

      // card selected style
      $('.library-card-selected').removeClass('library-card-selected');
      $(this).addClass('library-card-selected');

      $('.library-select-btn').addClass('library-select-btn-activated');
    })
    // Double click auto selects task
    .on('dblclick', function(event) {
      event.preventDefault();
      $('.library-select-btn').click();
    });

  // Label/title of method
  var label = $('<h4/>')
    .addClass('card-label')
    .html(task.label);

  // Structure elements appropriately
  label.appendTo(card);
  card.appendTo($('.library-left-panel-inner'));
}
